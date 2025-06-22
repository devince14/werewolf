from competition_template.werewolf_env.werewolf_env import WerewolfEnv, Role
# <<<<<<< fvt4qy-codex/提供狼人杀ai竞赛代码提交及测评方法
from competition_template.agents.belief_agent import BeliefAgent
# =======
from competition_template.agents_user.random_agent import RandomAgent
# >>>>>>> main

# roles: 1 wolf, 1 villager, 1 seer
roles = [Role.WOLF, Role.VILLAGER, Role.SEER]

def main():
    env = WerewolfEnv(roles, talk_history_len=10, max_nights=3)
    agents = {}
    for i, r in enumerate(roles):
# <<<<<<< fvt4qy-codex/提供狼人杀ai竞赛代码提交及测评方法
        agents[str(i)] = BeliefAgent(i, len(roles), r)
=======
        agents[str(i)] = RandomAgent(i, len(roles), r)
# >>>>>>> main
        env.add_agent(agents[str(i)])

    obs, _ = env.reset()
    done = False
    while not done:
        actions = {pid: a.act(env) for pid, a in agents.items()}
        obs, _, term, _, _ = env.step(actions)
        done = any(term.values())

    env.render(n_events=50, god=True)

if __name__ == "__main__":
    main()
