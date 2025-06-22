"""Minimal evaluation script for the Biweekly Cup."""
import importlib
import argparse
from typing import Type
from competition_template.werewolf_env.werewolf_env import WerewolfEnv, Role
from competition_template.agents.belief_agent import BeliefAgent
from competition_template.agents_user.random_agent import RandomAgent
import random


def load_agent(path: str) -> Type:
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

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

def evaluate(agent_cls: Type, n_games: int = 10000) -> float:
    roles = [Role.WOLF, Role.VILLAGER,Role.VILLAGER,Role.VILLAGER, Role.SEER]
    # 评估环境为: 1狼, 3村民, 1预言家
    # roles = assign_roles(1, 3)
    wins = 0
    for i in range(n_games):
        env = WerewolfEnv(roles, talk_history_len=10, max_nights=20)
        agents = {str(j): BeliefAgent(j, len(roles), r) for j, r in enumerate(roles)}

        # agents = {str(j): RandomAgent(j, len(roles), r) for j, r in enumerate(roles)}
        # player 0 will be replaced by contestant's agent
        agents['0'] = agent_cls(0, len(roles), roles[0])
        for a in agents.values():
            env.add_agent(a)
        obs, _ = env.reset()
        done = False
        while not done:
            actions = {pid: a.act(env) for pid, a in agents.items()}
            obs, _, term, _, _ = env.step(actions)
            done = any(term.values())
        winner = env._check_win()
        if winner == "WOLF":
            wins += 1
    return wins / n_games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", help="Import path to agent class")
    parser.add_argument("--games", type=int, default=10000, help="Number of games")
    args = parser.parse_args()
    agent_cls = load_agent(args.agent)
    win_rate = evaluate(agent_cls, args.games)
    print(f"Wolf win rate over {args.games} games: {win_rate:.2%}")


if __name__ == "__main__":
    main()
