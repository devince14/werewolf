"""Minimal evaluation script for the Biweekly Cup."""
import importlib
import argparse
from typing import Type
from competition_template.werewolf_env.werewolf_env import WerewolfEnv, Role
from competition_template.agents_user.random_agent import RandomAgent


def load_agent(path: str) -> Type:
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def evaluate(agent_cls: Type, n_games: int = 10) -> float:
    roles = [Role.WOLF, Role.VILLAGER, Role.SEER]
    wins = 0
    for i in range(n_games):
        env = WerewolfEnv(roles, talk_history_len=10, max_nights=3)
        agents = {str(j): RandomAgent(j, len(roles), r) for j, r in enumerate(roles)}
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
    parser.add_argument("--games", type=int, default=10, help="Number of games")
    args = parser.parse_args()
    agent_cls = load_agent(args.agent)
    win_rate = evaluate(agent_cls, args.games)
    print(f"Wolf win rate over {args.games} games: {win_rate:.2%}")


if __name__ == "__main__":
    main()
