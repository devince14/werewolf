import torch
import torch.optim as optim
import numpy as np
from collections import deque
import os
from werewolf_env_posg import Role, WerewolfEnv
from agents.posg_agent import BeliefAgent  # 假设之前的Agent代码保存在此文件中
import random
import torch.nn.functional as F
class WerewolfTrainer:
    def __init__(self, roles, num_episodes=1000, batch_size=32, lr=1e-4, gamma=0.99):
        """
        狼人杀训练器
        
        参数:
            roles: 角色分配列表
            num_episodes: 训练轮数
            batch_size: 批次大小
            lr: 学习率
            gamma: 折扣因子
        """
        self.roles = roles
        self.N = len(roles)
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        
        # 初始化环境
        self.env = WerewolfEnv(roles)
        
        # 初始化智能体
        self.agents = [BeliefAgent(i, self.N) for i in range(self.N)]
        
        # 优化器
        self.belief_optimizers = [
            optim.Adam(agent.belief_net.parameters(), lr=lr) 
            for agent in self.agents
        ]
        
        self.policy_optimizers = [
            optim.Adam(agent.policy_net.parameters(), lr=lr) 
            for agent in self.agents
        ]
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=10000)
        
        # 训练统计
        self.win_rates = {"GOOD": [], "WOLF": []}
        self.best_win_rate = 0
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def collect_experience(self, num_games=10):
        """收集经验数据"""
        for _ in range(num_games):
            obs, _ = self.env.reset()
            done = False
            episode_data = {str(i): [] for i in range(self.N)}
            
            while not done:
                actions = {}
                stage = 0 if self.env.stage == "talk" else 1 if self.env.stage == "vote" else 2
                
                for agent_id, agent in enumerate(self.agents):
                    agent_str = str(agent_id)
                    if obs[agent_str] is None:
                        continue
                    
                    # 更新历史观测
                    agent.update_history(obs[agent_str], stage)
                    
                    # 更新信念
                    agent.update_belief(
                        obs[agent_str],
                        obs[agent_str]['role'],  # 角色one-hot
                        stage
                    )
                    
                    # 学习已知信息
                    agent.learn_from_observation(obs[agent_str], self.env.event_log)
                    
                    # 获取动作和动作概率
                    action, log_prob, value = self._get_action_with_probs(
                        agent, obs[agent_str], stage
                    )
                    
                    # 存储当前状态
                    episode_data[agent_str].append({
                        'obs': obs[agent_str],
                        'action': action,
                        'log_prob': log_prob,
                        'value': value,
                        'stage': stage,
                        'alive_mask': obs[agent_str]['alive_mask'].copy()
                    })
                    
                    # 转换为环境期望的格式
                    if action.get('type') is not None:
                        if action['type'] == 'talk':
                            actions[agent_str] = np.array([0, action['role'], action['player']])
                        elif action['type'] == 'vote':
                            actions[agent_str] = np.array([action['target']])
                        elif action['type'] == 'kill':
                            actions[agent_str] = np.array([action['target']])
                        elif action['type'] == 'check':
                            actions[agent_str] = np.array([action['target']])
                
                # 环境步进
                next_obs, rewards, terms, _, _ = self.env.step(actions)
                done = all(terms.values())
                
                # 为每个智能体添加奖励
                for agent_id in range(self.N):
                    agent_str = str(agent_id)
                    if agent_str in episode_data and episode_data[agent_str]:
                        episode_data[agent_str][-1]['reward'] = rewards[agent_str]
                        episode_data[agent_str][-1]['done'] = terms[agent_str]
                
                obs = next_obs
            
            # 添加完整轨迹到回放缓冲区
            self.replay_buffer.append(episode_data)
            
            # 记录胜负
            winner = self.env.event_log[-1].get('winner')
            if winner in ["GOOD", "WOLF"]:
                self.win_rates[winner].append(1)
            else:
                # 平局或未结束，不计入统计
                pass
    
    def _get_action_with_probs(self, agent, obs, stage):
        """获取动作及其概率和状态值"""
        # 准备输入向量
        belief_flat = agent.belief.flatten().unsqueeze(0)
        obs_tensor = agent._encode_obs(obs, stage).unsqueeze(0)
        role_tensor = torch.tensor(obs['role']).float().unsqueeze(0)
        alive_tensor = torch.tensor(obs['alive_mask']).float().unsqueeze(0)
        stage_tensor = torch.tensor([stage]).float().unsqueeze(0)
        
        # 拼接输入
        policy_input = torch.cat([
            belief_flat,
            obs_tensor,
            role_tensor,
            alive_tensor,
            F.one_hot(torch.tensor([stage]), num_classes=3).float()
        ], dim=1).to(agent.device)
        
        # 角色类型 (0:狼人, 1:平民, 2:预言家)
        role_type = torch.argmax(role_tensor).item()
        
        # 决策网络预测
        with torch.no_grad():
            action_probs = agent.policy_net(
                policy_input, 
                alive_tensor.to(agent.device),
                stage,
                role_type
            )
        
        # 采样动作并获取对数概率
        if action_probs is None:
            return {}, None, None
        
        if stage == 0:  # 发言阶段
            claim_player_probs, claim_role_probs = action_probs
            player_choice = torch.multinomial(claim_player_probs, 1).item()
            role_choice = torch.multinomial(claim_role_probs, 1).item()
            
            # 计算对数概率
            log_prob = (
                torch.log(claim_player_probs[0, player_choice]) +
                torch.log(claim_role_probs[0, role_choice])
            )
            
            action = {'type': 'talk', 'player': player_choice, 'role': role_choice}
            value = 0  # 发言阶段无状态值估计
        
        elif stage == 1:  # 投票阶段
            vote_choice = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[0, vote_choice])
            action = {'type': 'vote', 'target': vote_choice}
            value = 0  # 投票阶段无状态值估计
        
        elif stage == 2:  # 夜间阶段
            if role_type == 0:  # 狼人
                kill_choice = torch.multinomial(action_probs, 1).item()
                log_prob = torch.log(action_probs[0, kill_choice])
                action = {'type': 'kill', 'target': kill_choice}
                value = 0
            elif role_type == 2:  # 预言家
                check_choice = torch.multinomial(action_probs, 1).item()
                log_prob = torch.log(action_probs[0, check_choice])
                action = {'type': 'check', 'target': check_choice}
                value = 0
            else:
                return {}, None, None
        
        return action, log_prob, value
    
    def update_networks(self):
        """更新信念网络和策略网络"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 随机采样批次
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # 更新每个智能体的网络
        for agent_id, agent in enumerate(self.agents):
            belief_losses = []
            policy_losses = []
            
            for episode_data in batch:
                agent_data = episode_data.get(str(agent_id), [])
                if not agent_data:
                    continue
                
                # 计算折扣回报
                rewards = [step['reward'] for step in agent_data]
                discounted_rewards = []
                cumulative_reward = 0
                for r in reversed(rewards):
                    cumulative_reward = r + self.gamma * cumulative_reward
                    discounted_rewards.insert(0, cumulative_reward)
                
                # 更新信念网络 (监督学习)
                for step in agent_data:
                    if 'done' in step and step['done']:
                        pass_sts = True
                        continue
                    else:
                        pass_sts = False
                    # 准备输入
                    obs = step['obs']
                    stage = step['stage']
                    role_onehot = obs['role']
                    prev_belief = agent.belief.clone()
                    
                    # 真实身份标签 (使用环境中的真实身份)
                    true_roles = self.env.roles
                    true_labels = torch.zeros(len(true_roles), 3)
                    for i, role in enumerate(true_roles):
                        if role == Role.WOLF:
                            true_labels[i, 0] = 1
                        elif role == Role.VILLAGER:
                            true_labels[i, 1] = 1
                        elif role == Role.SEER:
                            true_labels[i, 2] = 1
                    
                    # 更新信念
                    obs_tensor = agent._encode_obs(obs, stage).unsqueeze(0).to(agent.device)
                    role_tensor = torch.tensor(role_onehot).float().unsqueeze(0).to(agent.device)
                    prev_belief_tensor = prev_belief.unsqueeze(0).to(agent.device)
                    
                    # 准备历史序列
                    if len(agent.history_obs) > 0:
                        history_tensor = torch.stack(
                            agent.history_obs[-agent.max_history_len:]
                        ).unsqueeze(0).to(agent.device)
                    else:
                        history_tensor = torch.zeros(1, 1, obs_tensor.size(1) + 
                                                   prev_belief_tensor.size(2) + 
                                                   role_tensor.size(1)).to(agent.device)
                    
                    # 前向传播
                    new_belief = agent.belief_net(
                        history_tensor, prev_belief_tensor, role_tensor
                    )
                    
                    true_labels_index = torch.argmax(true_labels, dim=1).to(agent.device)
                    
                    # 确保new_belief的形状是(batch_size * num_players, num_classes)
                    new_belief_flat = new_belief.view(-1, 3)
                    
                    # 确保标签的形状是(batch_size * num_players)
                    true_labels_flat = true_labels_index.view(-1)
                    
                    loss = F.cross_entropy(
                        new_belief_flat, 
                        true_labels_flat
                    )
                    belief_losses.append(loss)
                
                # 更新策略网络 (强化学习)
                for idx, step in enumerate(agent_data):
                    # 计算优势函数
                    if step['value'] is None:
                        continue
                    else:
                        advantage = discounted_rewards[idx] - step['value']

                        # 策略损失
                        policy_loss = -step['log_prob'] * advantage
                        policy_losses.append(policy_loss)
            
            if not pass_sts:
                # 更新信念网络
                if belief_losses:
                    belief_loss = torch.stack(belief_losses).mean()
                    self.belief_optimizers[agent_id].zero_grad()
                    belief_loss.backward()
                    self.belief_optimizers[agent_id].step()
                
                # 更新策略网络
                if policy_losses:
                    policy_loss = torch.stack(policy_losses).mean()
                    self.policy_optimizers[agent_id].zero_grad()
                    policy_loss.backward()
                    self.policy_optimizers[agent_id].step()
    
    def evaluate(self, num_games=100):
        """评估当前模型性能"""
        win_counts = {"GOOD": 0, "WOLF": 0, "DRAW": 0}
        
        for _ in range(num_games):
            obs, _ = self.env.reset()
            done = False
            
            while not done:
                actions = {}
                stage = 0 if self.env.stage == "talk" else 1 if self.env.stage == "vote" else 2
                
                for agent_id, agent in enumerate(self.agents):
                    agent_str = str(agent_id)
                    if obs[agent_str] is None:
                        continue
                    
                    # 更新历史观测
                    agent.update_history(obs[agent_str], stage)
                    
                    # 更新信念
                    agent.update_belief(
                        obs[agent_str],
                        obs[agent_str]['role'],  # 角色one-hot
                        stage
                    )
                    
                    # 获取动作
                    action = agent.get_action(
                        obs[agent_str],
                        obs[agent_str]['role'],
                        stage,
                        obs[agent_str]['alive_mask']
                    )
                    
                    if action:
                        # 转换为环境期望的格式
                        if action['type'] == 'claim':
                            actions[agent_str] = np.array([0, action['role'], action['player']])
                        elif action['type'] == 'vote':
                            actions[agent_str] = np.array([action['target']])
                        elif action['type'] == 'kill':
                            actions[agent_str] = np.array([action['target']])
                        elif action['type'] == 'check':
                            actions[agent_str] = np.array([action['target']])
                
                # 环境步进
                obs, rewards, terms, _, _ = self.env.step(actions)
                done = all(terms.values())
            
            # 记录结果
            winner = self.env.event_log[-1].get('winner', 'DRAW')
            win_counts[winner] += 1
        
        # 计算胜率
        good_win_rate = win_counts["GOOD"] / num_games
        wolf_win_rate = win_counts["WOLF"] / num_games
        
        return good_win_rate, wolf_win_rate
    
    def train(self):
        """主训练循环"""
        print("开始训练狼人杀智能体...")
        print(f"配置: {self.N}人局 ({sum(1 for r in self.roles if r == Role.WOLF)}狼人)")
        
        for episode in range(self.num_episodes):
            # 收集经验
            self.collect_experience(num_games=5)
            
            # 更新网络
            self.update_networks()
            
            # 定期评估
            if episode % 100 == 0:
                good_win_rate, wolf_win_rate = self.evaluate(num_games=50)
                print(f"Episode {episode}: 好人胜率={good_win_rate:.2f}, 狼人胜率={wolf_win_rate:.2f}")
                
                # 保存最佳模型
                total_win_rate = (good_win_rate + wolf_win_rate) / 2
                if total_win_rate > self.best_win_rate:
                    self.best_win_rate = total_win_rate
                    self.save_models(f"best_model_ep{episode}_win{total_win_rate:.2f}")
        
        print("训练完成!")
        self.save_models("final_model")
    
    def save_models(self, model_name):
        """保存模型"""
        save_dir = os.path.join(self.model_dir, model_name)
        os.makedirs(save_dir, exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            # 保存信念网络
            torch.save(agent.belief_net.state_dict(), 
                      os.path.join(save_dir, f"agent_{i}_belief_net.pth"))
            
            # 保存策略网络
            torch.save(agent.policy_net.state_dict(), 
                      os.path.join(save_dir, f"agent_{i}_policy_net.pth"))
        
        print(f"模型已保存到: {save_dir}")
    
    def load_models(self, model_dir):
        """加载模型"""
        for i, agent in enumerate(self.agents):
            # 加载信念网络
            belief_path = os.path.join(model_dir, f"agent_{i}_belief_net.pth")
            if os.path.exists(belief_path):
                agent.belief_net.load_state_dict(torch.load(belief_path))
            
            # 加载策略网络
            policy_path = os.path.join(model_dir, f"agent_{i}_policy_net.pth")
            if os.path.exists(policy_path):
                agent.policy_net.load_state_dict(torch.load(policy_path))
        
        print(f"从 {model_dir} 加载模型")

# 训练配置
if __name__ == "__main__":
    # 示例角色分配：6人局 (2狼人, 3平民, 1预言家)
    roles = [Role.VILLAGER, Role.VILLAGER, Role.WOLF, Role.SEER, Role.WOLF, Role.VILLAGER]
    
    # 创建训练器
    trainer = WerewolfTrainer(
        roles=roles,
        num_episodes=5000,  # 训练轮数
        batch_size=32,       # 批次大小
        lr=1e-4,            # 学习率
        gamma=0.95           # 折扣因子
    )
    
    # 开始训练
    trainer.train()
    
    # 最终评估
    good_win_rate, wolf_win_rate = trainer.evaluate(num_games=100)
    print(f"最终评估: 好人胜率={good_win_rate:.2f}, 狼人胜率={wolf_win_rate:.2f}")