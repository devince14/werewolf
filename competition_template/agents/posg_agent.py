import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from werewolf_env_posg import Role, WerewolfEnv, PHASE_TALK, TalkType

class BeliefUpdateNet(nn.Module):
    """认知网络：根据历史观测更新玩家身份信念"""
    def __init__(self, input_dim, hidden_dim, N, lstm_layers=1):
        super().__init__()
        self.N = N
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim + 3*N + 3, 3*N)  # LSTM输出 + 信念展平 + 身份
        
    def forward(self, history_obs, prev_belief, role):
        # history_obs: (batch, seq_len, input_dim)
        # prev_belief: (batch, N, 3)
        # role: (batch, 3)
        
        # LSTM处理历史序列
        lstm_out, _ = self.lstm(history_obs)
        h_t = lstm_out[:, -1, :]  # 取序列最后一个输出
        
        # 拼接LSTM输出、上一时刻信念和身份
        prev_belief_flat = prev_belief.view(prev_belief.size(0), -1)
        combined = torch.cat([h_t, prev_belief_flat, role], dim=1)
        
        # 生成新信念
        output = self.fc(combined)
        new_belief = output.view(prev_belief.size(0), self.N, 3)
        new_belief = F.softmax(new_belief, dim=2)
        
        return new_belief

class PolicyNet(nn.Module):
    """决策网络：根据当前信念和观测选择动作"""
    def __init__(self, input_dim, hidden_dim, N):
        super().__init__()
        self.N = N
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 定义各动作头
        self.vote_head = nn.Linear(hidden_dim, N+1)
        self.claim_player = nn.Linear(hidden_dim, N)
        self.claim_role = nn.Linear(hidden_dim, 3)
        self.kill_head = nn.Linear(hidden_dim, N+1)
        self.check_head = nn.Linear(hidden_dim, N+1)
        
    def forward(self, x, alive_mask, stage, role_type):
        """
        x: 输入向量 [信念, 观测, 身份, 存活掩码, 阶段]
        alive_mask: (batch, N)
        stage: 阶段标识 (0:talk, 1:vote, 2:night)
        role_type: 角色标识 (0:wolf, 1:villager, 2:seer)
        """
        x = self.shared_fc(x)
        
        # 投票动作头
        vote_logits = self.vote_head(x)
        vote_probs = self._masked_softmax(vote_logits, alive_mask, has_abstain=True)
        
        # 声称玩家选择头
        claim_player_logits = self.claim_player(x)
        claim_player_probs = self._masked_softmax(claim_player_logits, alive_mask)
        
        # 声称角色选择头
        claim_role_logits = self.claim_role(x)
        claim_role_probs = F.softmax(claim_role_logits, dim=1)
        
        # 狼人击杀头
        kill_logits = self.kill_head(x)
        kill_probs = self._masked_softmax(kill_logits, alive_mask, has_abstain=True)
        
        # 预言家查验头
        check_logits = self.check_head(x)
        check_probs = self._masked_softmax(check_logits, alive_mask, has_abstain=True)
        
        # 根据阶段和角色选择输出
        if stage == 0:  # 白天发言阶段
            return claim_player_probs, claim_role_probs
        elif stage == 1:  # 白天投票阶段
            return vote_probs
        elif stage == 2:  # 夜间阶段
            if role_type == 0:  # 狼人
                return kill_probs
            elif role_type == 2:  # 预言家
                return check_probs
        return None
    
    def _masked_softmax(self, logits, mask, has_abstain=False):
        """应用存活掩码的softmax"""
        if has_abstain:
            # 前N个元素对应玩家，最后一个元素是弃权/不操作
            player_mask = torch.cat([mask, torch.ones(mask.size(0), 1).to(mask.device)], dim=1)
        else:
            player_mask = mask
            
        # 将掩码为0的位置设置为负无穷
        masked_logits = logits.masked_fill(player_mask == 0, float('-inf'))
        return F.softmax(masked_logits, dim=1)

class BeliefAgent:
    """基于信念网络和决策网络的狼人杀Agent"""
    def __init__(self, agent_id, N, device='cpu'):
        self.id = agent_id
        self.N = N
        self.device = device
        
        # 网络参数
        obs_dim = 3 + N + 5  # 阶段(3) + 存活掩码(N) + 最近事件(5)
        belief_dim = 3 * N
        input_dim_belief = obs_dim + belief_dim + 3  # 观测 + 信念 + 身份
        input_dim_policy = belief_dim + obs_dim + 3 + N + 3  # 信念 + 观测 + 身份 + 存活掩码 + 阶段
        
        # 初始化网络
        self.belief_net = BeliefUpdateNet(input_dim_belief, 128, N).to(device)
        self.policy_net = PolicyNet(input_dim_policy, 256, N).to(device)
        
        # 初始化信念状态
        self.reset_beliefs()
        
        # 存储历史观测
        self.history_obs = []
        self.max_history_len = 20
        
    def reset_beliefs(self):
        """初始化信念状态"""
        # 初始信念：均匀分布
        self.belief = torch.ones(self.N, 3) / 3
        
        # 已知信息
        self.known_wolves = []  # 狼人同伴
        self.seer_checks = {}   # 预言家查验结果 {player_id: role}
    
    def update_belief(self, obs, role_onehot, stage):
        """使用认知网络更新信念"""
        # 将观测编码为向量
        obs_tensor = self._encode_obs(obs, stage).unsqueeze(0).to(self.device)
        role_tensor = torch.tensor(role_onehot).float().unsqueeze(0).to(self.device)
        prev_belief = self.belief.unsqueeze(0).to(self.device)
        
        # 准备历史序列
        if len(self.history_obs) > 0:
            history_tensor = torch.stack(self.history_obs[-self.max_history_len:]).unsqueeze(0)
        else:
            history_tensor = torch.zeros(1, 1, obs_tensor.size(1) + prev_belief.size(2) + role_tensor.size(1)).to(self.device)
        
        # 认知网络更新信念
        with torch.no_grad():
            new_belief = self.belief_net(history_tensor, prev_belief, role_tensor)
            new_belief = new_belief.squeeze(0).cpu()
        
        # 应用已知信息修正
        if role_onehot[0] == 1:  # 狼人
            for wolf_id in self.known_wolves:
                new_belief[wolf_id] = torch.tensor([0.0, 1.0, 0.0])  # 确定狼人身份
        elif role_onehot[2] == 1:  # 预言家
            for player_id, role_val in self.seer_checks.items():
                role_vec = torch.zeros(3)
                role_vec[role_val] = 1.0
                new_belief[player_id] = role_vec  # 用查验结果覆盖
        
        self.belief = new_belief
        return new_belief
    
    def get_action(self, obs, role_onehot, stage, alive_mask):
        """使用决策网络选择动作"""
        # 准备输入向量
        belief_flat = self.belief.flatten().unsqueeze(0)
        obs_tensor = self._encode_obs(obs, stage).unsqueeze(0)
        role_tensor = torch.tensor(role_onehot).float().unsqueeze(0)
        alive_tensor = torch.tensor(alive_mask).float().unsqueeze(0)
        stage_tensor = torch.tensor([stage]).float().unsqueeze(0)
        
        # 拼接输入
        policy_input = torch.cat([
            belief_flat,
            obs_tensor,
            role_tensor,
            alive_tensor,
            F.one_hot(torch.tensor([stage]), num_classes=3).float()
        ], dim=1).to(self.device)
        
        # 角色类型 (0:狼人, 1:平民, 2:预言家)
        role_type = torch.argmax(role_tensor).item()
        
        # 决策网络预测
        with torch.no_grad():
            action_probs = self.policy_net(
                policy_input, 
                alive_tensor.to(self.device),
                stage,
                role_type
            )
        
        # 采样动作
        if action_probs is None:
            return None  # 无动作
        
        if stage == 0:  # 发言阶段
            claim_player_probs, claim_role_probs = action_probs
            player_choice = torch.multinomial(claim_player_probs, 1).item()
            role_choice = torch.multinomial(claim_role_probs, 1).item()
            return {'type': 'talk', 'player': player_choice, 'role': role_choice}
        
        elif stage == 1:  # 投票阶段
            vote_choice = torch.multinomial(action_probs, 1).item()
            return {'type': 'vote', 'target': vote_choice}
        
        elif stage == 2:  # 夜间阶段
            if role_type == 0:  # 狼人
                kill_choice = torch.multinomial(action_probs, 1).item()
                return {'type': 'kill', 'target': kill_choice}
            elif role_type == 2:  # 预言家
                check_choice = torch.multinomial(action_probs, 1).item()
                return {'type': 'check', 'target': check_choice}
        
        return None
    
    def _encode_obs(self, obs, stage):
        """将环境观测编码为向量"""
        # 阶段one-hot (talk, vote, night)
        stage_oh = torch.zeros(3)
        stage_oh[stage] = 1.0
        
        # 存活掩码
        alive_mask = torch.tensor(obs['alive_mask'])
        
        # 最近事件 (取历史最后一条)
        talk_history = obs['talk_history']
        last_event = torch.zeros(5)
        if len(talk_history) > 0 and not np.all(talk_history[-1] == -1):
            last_event = torch.tensor(talk_history[-1])
        
        # 拼接所有部分
        return torch.cat([stage_oh, alive_mask, last_event])
    
    def update_history(self, obs, stage):
        """更新历史观测序列"""
        # 编码当前观测
        obs_encoded = self._encode_obs(obs, stage)
        
        # 添加角色信息和上一时刻信念
        role_oh = torch.zeros(3)
        if obs['role'][0] == 1: role_oh[0] = 1  # 狼人
        elif obs['role'][1] == 1: role_oh[1] = 1  # 平民
        elif obs['role'][2] == 1: role_oh[2] = 1  # 预言家
        
        # 准备认知网络输入
        belief_flat = self.belief.flatten()
        combined = torch.cat([obs_encoded, belief_flat, role_oh])
        
        # 添加到历史
        self.history_obs.append(combined)
        if len(self.history_obs) > self.max_history_len:
            self.history_obs.pop(0)
    
    def learn_from_observation(self, obs, event_log):
        """从观测中学习已知信息（狼人同伴、查验结果）"""
        # 狼人：首日获知所有狼人身份
        if (obs['role'][0] == 1 and  # 是狼人
            len(self.history_obs) == 0):  # 首日
            self.known_wolves = [i for i, role in enumerate(obs['role_assignments']) 
                                if role == Role.WOLF and i != self.id]
        
        # 预言家：记录查验结果
        if obs['role'][2] == 1:  # 是预言家
            for event in event_log:
                if event.get('phase') == 'seer_check' and event.get('seer') == self.id:
                    target = event['target']
                    role_val = event['role']
                    self.seer_checks[target] = role_val

    def act(self, env):
        obs = env.get_obs()
        stage = 0 if env.stage == "talk" else 1 if env.stage == "vote" else 2
        agent_id = str(self.id)
        action = self.get_action(
            obs[agent_id],
            obs[agent_id]['role'],
            stage,
            obs[agent_id]['alive_mask']
        )
        if action:
            # 转换为环境期望的格式
            if action['type'] == 'talk':
                agent_id = int(agent_id)
                if env.roles[agent_id] != Role.SEER:
                    claim_seer = 0
                else:
                    claim_seer = 1
                res = np.array([claim_seer, action['role'], action['player']])
                record_talk(env, agent_id, claim_seer, action['role'], action['player'])
            elif action['type'] == 'vote':
                res = np.array([action['target']])
            elif action['type'] == 'kill':
                res = np.array([action['target']])
            elif action['type'] == 'check':
                res = np.array([action['target']])
        
            return res
        else:
            return

def record_talk(env, agent_id, claim_seer: int, talk_type: int, target: int):
    """记录发言到环境日志"""
    if target == env.N:
        target = -1

    def alive_list(): 
        return [i for i,a in enumerate(env.alive) if a]
        
    # 检查是否是遗言阶段
    # 只有第一夜死亡的玩家才能留遗言
    is_first_night_legacy = (not env.alive[agent_id] and env.day == 1)
    # print(f"[DEBUG] 遗言检测: agent_id={agent_id}, alive={env.alive[agent_id]}, day={env.day}, is_legacy={is_first_night_legacy}")
                                
    if is_first_night_legacy:
        print(f"[DEBUG] 玩家 {agent_id} 进入遗言模式（第一夜死亡）")
        # 如果是遗言阶段，使用专门的遗言记录函数
        _record_legacy(agent_id, env, talk_type, target)
        return

    # 处理预言家身份声明
    if claim_seer == 1:
        # 记录声称预言家
        log_entry = (PHASE_TALK, agent_id, int(TalkType.CLAIM_SEER), -1, -1)
        env.public_log.append(log_entry)
        env.event_log.append({
            "day": env.day,
            "phase": "talk",
            "speaker": agent_id,
            "type": int(TalkType.CLAIM_SEER),
            "target": -1,
            "alive": alive_list()
        })
        
        # 立即更新所有玩家的信念（声称预言家）
        for agent in env.agents:
            if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                agent.strategies['belief_update'].execute(env, [log_entry])
        #print(env.seer_records)
        # 如果是真预言家且有验人结果，立即公布
        if env.roles[agent_id] == Role.SEER and env.seer_records:
            # 第二天首次跳身份时，公布所有验人结果
            if env.day == 2:
                # 收集所有需要公布的验人结果
                results_to_report = []
                for day, tgt, role_int in env.seer_records:
                    if day < env.day:  # 只报告之前天数的结果
                        results_to_report.append((day, tgt, role_int))
                
                # 按优先级排序：村民结果优先，然后是狼人结果
                wolf_results = [(d, t, r) for d, t, r in results_to_report if r == Role.WOLF]
                villager_results = [(d, t, r) for d, t, r in results_to_report if r == Role.VILLAGER]
                
                # 优先公布村民结果
                for day, tgt, role_int in villager_results:
                    talktype = TalkType.SUPPORT
                    log_entry = (PHASE_TALK, agent_id, int(talktype), tgt, role_int)
                    env.public_log.append(log_entry)
                    env.event_log.append({
                        "day": int(env.day),
                        "phase": "talk",
                        "speaker": agent_id,
                        "type": int(talktype),
                        "target": tgt,
                        "role": role_int,
                        "check_day": day,  # 添加查验的天数
                        "alive": alive_list()
                    })
                    
                    # 立即更新所有玩家的信念（验人结果）
                    for agent in env.agents:
                        if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                            agent.strategies['belief_update'].execute(env, [log_entry])
                
                # 然后公布狼人结果
                for day, tgt, role_int in wolf_results:
                    talktype = TalkType.ACCUSE
                    log_entry = (PHASE_TALK, agent_id, int(talktype), tgt, role_int)
                    env.public_log.append(log_entry)
                    env.event_log.append({
                        "day": int(env.day),
                        "phase": "talk",
                        "speaker": agent_id,
                        "type": int(talktype),
                        "target": tgt,
                        "role": role_int,
                        "check_day": day,  # 添加查验的天数
                        "alive": alive_list()
                    })
                    
                    # 立即更新所有玩家的信念（验人结果）
                    for agent in env.agents:
                        if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                            agent.strategies['belief_update'].execute(env, [log_entry])
            
            # 其他天数，只公布最新的验人结果
            elif env.seer_records[-1][0] == env.day - 1:
                _, tgt, role_int = env.seer_records[-1]
                talktype = TalkType.ACCUSE if role_int == Role.WOLF else TalkType.SUPPORT
                log_entry = (PHASE_TALK, agent_id, int(talktype), tgt, role_int)
                env.public_log.append(log_entry)
                env.event_log.append({
                    "day": int(env.day),
                    "phase": "talk",
                    "speaker": agent_id,
                    "type": int(talktype),
                    "target": tgt,
                    "role": role_int,
                    "check_day": env.day - 1,  # 添加查验的天数
                    "alive": alive_list()
                })
                
                # 立即更新所有玩家的信念（验人结果）
                for agent in env.agents:
                    if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                        agent.strategies['belief_update'].execute(env, [log_entry])
        
        # 如果是狼人假跳预言家，且有指控或支持目标，立即执行
        elif env.roles[agent_id] == Role.WOLF and talk_type in (TalkType.ACCUSE, TalkType.SUPPORT) and target != -1:
            # 根据talk_type设置role_int：ACCUSE为0，SUPPORT为1
            role_int = 1 if talk_type == TalkType.SUPPORT else 0
            log_entry = (PHASE_TALK, agent_id, int(talk_type), target, role_int)
            env.public_log.append(log_entry)
            env.event_log.append({
                "day": env.day,
                "phase": "talk",
                "speaker": agent_id,
                "type": int(talk_type),
                "target": target,
                "role": role_int,
                "alive": alive_list()
            })
            
            # 立即更新所有玩家的信念（狼人的指控/支持）
            for agent in env.agents:
                if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                    agent.strategies['belief_update'].execute(env, [log_entry])
    
    # 处理普通发言
    elif talk_type == TalkType.CLAIM_GOOD:
        log_entry = (PHASE_TALK, agent_id, int(TalkType.CLAIM_GOOD), -1, -1)
        env.public_log.append(log_entry)
        env.event_log.append({
            "day": env.day,
            "phase": "talk",
            "speaker": agent_id,
            "type": int(TalkType.CLAIM_GOOD),
            "target": -1,
            "alive": alive_list()
        })
        
        # 立即更新所有玩家的信念
        for agent in env.agents:
            if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                agent.strategies['belief_update'].execute(env, [log_entry])
                
    elif talk_type in (TalkType.ACCUSE, TalkType.SUPPORT):
        # 如果是预言家的验人结果，记录role_int
        role_int = -1
        if (env.roles[agent_id] == Role.SEER and env.seer_records 
            and env.seer_records[-1][0] == env.day - 1 
            and target == env.seer_records[-1][1]):
            role_int = env.seer_records[-1][2]
            
        log_entry = (PHASE_TALK, agent_id, int(talk_type), target, role_int)
        env.public_log.append(log_entry)
        env.event_log.append({
            "day": env.day,
            "phase": "talk",
            "speaker": agent_id,
            "type": int(talk_type),
            "target": target,
            "role": role_int if role_int != -1 else None,
            "alive": alive_list()
        })
        
        # 立即更新所有玩家的信念
        for agent in env.agents:
            if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                agent.strategies['belief_update'].execute(env, [log_entry])

def _record_legacy(agent_id, env, talk_type: int, target: int):
    """记录遗言事件"""
    print(f"[DEBUG] _record_legacy被调用: agent_id={agent_id}, talk_type={talk_type}, target={target}")
    
    # 如果是真预言家且有验人结果，优先报告查验结果并亮身份
    if env.roles[agent_id] == Role.SEER and hasattr(env, 'seer_records') and env.seer_records:
        # 先声称预言家身份
        claim_entry = (PHASE_TALK, agent_id, int(TalkType.CLAIM_SEER), -1, -1)
        env.public_log.append(claim_entry)
        env.event_log.append({
            "day": env.day,
            "phase": "legacy",
            "speaker": agent_id,
            "type": int(TalkType.CLAIM_SEER),
            "target": -1,
            "alive": [i for i,a in enumerate(env.alive) if a]
        })
        
        # 报告所有查验结果
        for day, tgt, role_int in env.seer_records:
            if day < env.day:  # 只报告之前天数的结果
                result_talk_type = TalkType.ACCUSE if role_int == Role.WOLF else TalkType.SUPPORT
                log_entry = (PHASE_TALK, agent_id, int(result_talk_type), tgt, role_int)
                env.public_log.append(log_entry)
                env.event_log.append({
                    "day": env.day,
                    "phase": "legacy",
                    "speaker": agent_id,
                    "type": int(result_talk_type),
                    "target": tgt,
                    "role": role_int,
                    "check_day": day,
                    "alive": [i for i,a in enumerate(env.alive) if a]
                })
    else:
        # 非预言家按照正常发言策略
        log_entry = (PHASE_TALK, agent_id, int(talk_type), target, -1)
        env.public_log.append(log_entry)
        env.event_log.append({
            "day": env.day,
            "phase": "legacy",
            "speaker": agent_id,
            "type": int(talk_type),
            "target": target,
            "alive": [i for i,a in enumerate(env.alive) if a]
        })
    
    print(f"[DEBUG] 遗言事件已添加到事件日志")



