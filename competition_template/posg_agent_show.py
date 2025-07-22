from agents.posg_agent import BeliefAgent, record_talk
from agents.posg_agent_train import WerewolfTrainer
from werewolf_env_posg import Role, WerewolfEnv
import numpy as np
# 示例：使用信念Agent进行游戏
def play_game_with_belief_agents(roles, num_games=1, agents = None):
    env = WerewolfEnv(roles)
    if agents is None:
        agents = {str(i): BeliefAgent(i, len(roles)) for i in range(len(roles))}
    for game in range(num_games):
        obs, _ = env.reset()

        
        while env._check_win() == None:
            actions = {}
            stage = 0 if env.stage == "talk" else 1 if env.stage == "vote" else 2
            
            for agent_id, agent in agents.items():
                if obs[agent_id] is None:
                    continue
                # 更新历史观测
                agent.update_history(obs[agent_id], stage)
                
                # 更新信念
                agent.update_belief(
                    obs[agent_id],
                    obs[agent_id]['role'],  # 角色one-hot
                    stage
                )
                
                # 学习已知信息
                agent.learn_from_observation(obs[agent_id], env.event_log)
                
                # 获取动作
                action = agent.get_action(
                    obs[agent_id],
                    obs[agent_id]['role'],
                    stage,
                    obs[agent_id]['alive_mask']
                )
                if env.alive[int(agent_id)]:
                    if action:
                        # 转换为环境期望的格式
                        if action['type'] == 'talk':
                            agent_id = int(agent_id)
                            if env.roles[agent_id] != Role.SEER:
                                claim_seer = 0
                            else:
                                claim_seer = 1
                            actions[agent_id] = np.array([claim_seer, action['role'], action['player']])
                            record_talk(env, agent_id, claim_seer, action['role'], action['player'])
                        elif action['type'] == 'vote':
                            actions[agent_id] = np.array([action['target']])
                        elif action['type'] == 'kill':
                            actions[agent_id] = np.array([action['target']])
                        elif action['type'] == 'check':
                            actions[agent_id] = np.array([action['target']])
            # 环境步进
            obs, rewards, terms, truncs, infos = env.step(actions)
            # print(env.event_log)
            env.render(n_events=50, god=True)
    
    return env.event_log

if __name__=='__main__':

    roles = [Role.VILLAGER, Role.VILLAGER, Role.WOLF, Role.SEER, Role.WOLF, Role.VILLAGER]

    trainer = WerewolfTrainer(
        roles=roles,
    )

    trainer.load_models("posg_model")
    play_game_with_belief_agents(roles, agents={str(i): trainer.agents[i] for i in range(len(roles))})