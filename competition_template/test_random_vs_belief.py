"""
测试随机策略 vs 信念策略的表现差异
"""

import numpy as np
import random
from werewolf_env.werewolf_env import WerewolfEnv, Role, TalkType
from agents.belief_agent import BeliefAgent

class RandomAgent:
    """完全随机的智能体"""
    
    def __init__(self, agent_id: int, num_agents: int, role: Role):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.role = role

    def act(self, env):
        stage = env.stage
        
        if stage == "talk":
            if self.role == Role.SEER:
                return np.array([1, TalkType.CLAIM_SEER, env.N])
            else:
                return np.array([0, TalkType.CLAIM_GOOD, env.N])
            
        elif stage == "vote":
            alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
            if alive_players:
                return np.array([random.choice(alive_players)])
            return np.array([env.N])
            
        elif stage == "night":
            alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
            
            if self.role == Role.SEER:
                if alive_players:
                    return np.array([random.choice(alive_players)])
                return np.array([env.N])
            elif self.role == Role.WOLF:
                if alive_players:
                    return np.array([random.choice(alive_players)])
                return np.array([env.N])
            else:
                return np.array([env.N])
        
        return np.array([env.N])

def test_strategy_comparison(n_games=1000):
    """比较随机策略和信念策略"""
    
    # 统计不同角色的胜负情况
    random_stats = {
        Role.WOLF: {'wins': 0, 'total': 0},
        Role.SEER: {'wins': 0, 'total': 0},
        Role.VILLAGER: {'wins': 0, 'total': 0}
    }
    
    belief_stats = {
        Role.WOLF: {'wins': 0, 'total': 0},
        Role.SEER: {'wins': 0, 'total': 0},
        Role.VILLAGER: {'wins': 0, 'total': 0}
    }
    
    print(f"开始测试 {n_games} 场游戏...")
    
    for i in range(n_games):
        if i % 100 == 0:
            print(f"进度: {i}/{n_games}")
            
        # 随机分配角色
        roles = [Role.WOLF] + [Role.VILLAGER] * 3 + [Role.SEER]
        random.shuffle(roles)
        
        # 测试随机策略
        player_role = roles[0]
        random_stats[player_role]['total'] += 1
        
        env = WerewolfEnv(roles, talk_history_len=10, max_nights=20)
        agents = {str(j): BeliefAgent(j, len(roles), r) for j, r in enumerate(roles)}
        agents['0'] = RandomAgent(0, len(roles), roles[0])
        
        for a in agents.values():
            env.add_agent(a)
        obs, _ = env.reset()
        done = False
        while not done:
            actions = {pid: a.act(env) for pid, a in agents.items()}
            obs, _, term, _, _ = env.step(actions)
            done = any(term.values())
        winner = env._check_win()
        
        if (player_role == Role.WOLF and winner == "WOLF") or \
           (player_role != Role.WOLF and winner == "GOOD"):
            random_stats[player_role]['wins'] += 1
        
        # 测试信念策略
        belief_stats[player_role]['total'] += 1
        
        env = WerewolfEnv(roles, talk_history_len=10, max_nights=20)
        agents = {str(j): BeliefAgent(j, len(roles), r) for j, r in enumerate(roles)}
        
        for a in agents.values():
            env.add_agent(a)
        obs, _ = env.reset()
        done = False
        while not done:
            actions = {pid: a.act(env) for pid, a in agents.items()}
            obs, _, term, _, _ = env.step(actions)
            done = any(term.values())
        winner = env._check_win()
        
        if (player_role == Role.WOLF and winner == "WOLF") or \
           (player_role != Role.WOLF and winner == "GOOD"):
            belief_stats[player_role]['wins'] += 1
    
    # 打印结果
    print("\n=== 测试结果 ===")
    role_names = {Role.WOLF: "狼人", Role.SEER: "预言家", Role.VILLAGER: "村民"}
    
    print("\n随机策略:")
    for role in [Role.WOLF, Role.SEER, Role.VILLAGER]:
        stats = random_stats[role]
        if stats['total'] > 0:
            win_rate = stats['wins'] / stats['total']
            print(f"  {role_names[role]}: {stats['wins']}/{stats['total']} = {win_rate:.2%}")
    
    print("\n信念策略:")
    for role in [Role.WOLF, Role.SEER, Role.VILLAGER]:
        stats = belief_stats[role]
        if stats['total'] > 0:
            win_rate = stats['wins'] / stats['total']
            print(f"  {role_names[role]}: {stats['wins']}/{stats['total']} = {win_rate:.2%}")
    
    # 计算总体胜率
    random_total_wins = sum(stats['wins'] for stats in random_stats.values())
    random_total_games = sum(stats['total'] for stats in random_stats.values())
    random_overall = random_total_wins / random_total_games if random_total_games > 0 else 0
    
    belief_total_wins = sum(stats['wins'] for stats in belief_stats.values())
    belief_total_games = sum(stats['total'] for stats in belief_stats.values())
    belief_overall = belief_total_wins / belief_total_games if belief_total_games > 0 else 0
    
    print(f"\n总体胜率:")
    print(f"  随机策略: {random_total_wins}/{random_total_games} = {random_overall:.2%}")
    print(f"  信念策略: {belief_total_wins}/{belief_total_games} = {belief_overall:.2%}")
    print(f"  差异: {belief_overall - random_overall:.2%}")

if __name__ == "__main__":
    test_strategy_comparison(1000) 