"""Minimal evaluation script for the Biweekly Cup."""
import importlib
import argparse
import random
from typing import Type
from werewolf_env_posg.werewolf_env_posg import WerewolfEnv, Role
from agents.belief_agent import BeliefAgent
import sys
import os
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# def load_agent(path: str) -> Type:
#     module_name, class_name = path.rsplit(".", 1)
#     module = importlib.import_module(module_name)
#     return getattr(module, class_name)

def load_agent_posg(roles, model_dir="models/posg_model"):
    from agents.posg_agent_train import WerewolfTrainer
    trainer = WerewolfTrainer(roles)
    trainer.load_models(model_dir)
    return trainer.agents

def assign_roles(num_wolves: int, num_villagers: int):
    """
    随机分配角色
    
    Args:
        num_wolves: 狼人数量
        num_villagers: 村民数量（不包括预言家）
        
    Returns:
        List[Role]: 随机排列的角色列表
    """
    # 创建角色列表
    roles = [Role.WOLF] * num_wolves + [Role.VILLAGER] * num_villagers + [Role.SEER]
    
    # 随机打乱角色顺序
    random.shuffle(roles)
    
    return roles 

def evaluate(agent_eva, n_games: int = 10000) -> dict:
    """
    评估智能体性能
    
    Args:
        agent_cls: 参赛选手的智能体类
        n_games: 游戏场次数
        
    Returns:
        dict: 包含各角色胜率的字典
    """
    # 统计不同角色的胜负情况
    role_stats = {
        Role.WOLF: {'wins': 0, 'total': 0},
        Role.SEER: {'wins': 0, 'total': 0},
        Role.VILLAGER: {'wins': 0, 'total': 0}
    }
    
    for i in range(n_games):
        # 随机分配角色
        roles = assign_roles(2, 3)  # 1狼人, 3村民, 1预言家
        
        # 参赛选手分配到0号位置的角色
        player_role = roles[0]
        role_stats[player_role]['total'] += 1
        
        env = WerewolfEnv(roles, talk_history_len=10, max_nights=20)
        agents = {str(j): BeliefAgent(j, len(roles), r) for j, r in enumerate(roles)}

        # player 0 will be replaced by contestant's agent
        model_roles = [Role.VILLAGER, Role.VILLAGER, Role.WOLF, Role.SEER, Role.WOLF, Role.VILLAGER]
        if roles[0]==Role.VILLAGER:
            agents['0'] = agent_eva[0]
        elif roles[0]==Role.WOLF:
            agents['0'] = agent_eva[2]
        elif roles[0]==Role.SEER:
            agents['0'] = agent_eva[3]
        for a in agents.values():
            env.add_agent(a)
        
        obs, _ = env.reset()
        done = False
        while not done:
            ### posg 
            stage = 0 if env.stage == "talk" else 1 if env.stage == "vote" else 2
            if obs['0'] is None:
                continue
            # 更新历史观测
            agents['0'].update_history(obs['0'], stage)
            # 更新信念
            agents['0'].update_belief(
                obs['0'],
                obs['0']['role'],  # 角色one-hot
                stage
            )
            # 学习已知信息
            agents['0'].learn_from_observation(obs['0'], env.event_log)

            actions = {pid: agents[pid].act(env) for pid in list(agents.keys())}
            obs, _, term, _, _ = env.step(actions)
            done = any(term.values())
        winner = env._check_win()
        
        # 根据参赛选手的角色和游戏结果判断是否获胜
        if (player_role == Role.WOLF and winner == "WOLF") or \
           (player_role != Role.WOLF and winner == "GOOD"):
            role_stats[player_role]['wins'] += 1
    
    # 计算各角色胜率
    results = {}
    total_wins = 0
    total_games = 0
    
    for role in [Role.WOLF, Role.SEER, Role.VILLAGER]:
        if role_stats[role]['total'] > 0:
            win_rate = role_stats[role]['wins'] / role_stats[role]['total']
            results[role] = {
                'win_rate': win_rate,
                'wins': role_stats[role]['wins'],
                'total': role_stats[role]['total']
            }
            total_wins += role_stats[role]['wins']
            total_games += role_stats[role]['total']
        else:
            results[role] = {
                'win_rate': 0.0,
                'wins': 0,
                'total': 0
            }
    
    # 计算总胜率
    overall_win_rate = total_wins / total_games if total_games > 0 else 0.0
    results['overall'] = {
        'win_rate': overall_win_rate,
        'wins': total_wins,
        'total': total_games
    }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("agent", help="Import path to agent class")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    args = parser.parse_args()
    
    print(f"=== 智能体评测结果 (总计 {args.games} 场游戏) ===")
    print()
    
    model_roles = [Role.VILLAGER, Role.VILLAGER, Role.WOLF, Role.SEER, Role.WOLF, Role.VILLAGER]
    agent_eva = load_agent_posg(model_roles)
    results = evaluate(agent_eva, args.games)
    # 打印各角色的详细结果
    role_names = {
        Role.WOLF: "狼人",
        Role.SEER: "预言家", 
        Role.VILLAGER: "村民"
    }
    
    for role in [Role.WOLF, Role.SEER, Role.VILLAGER]:
        role_name = role_names[role]
        stats = results[role]
        if stats['total'] > 0:
            print(f"{role_name}: {stats['wins']}/{stats['total']} = {stats['win_rate']:.2%}")
        else:
            print(f"{role_name}: 0/0 = N/A")
    
    print()
    print(f"总体胜率: {results['overall']['wins']}/{results['overall']['total']} = {results['overall']['win_rate']:.2%}")


if __name__ == "__main__":
    main()
