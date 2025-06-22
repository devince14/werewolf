from competition_template.werewolf_env.werewolf_env import WerewolfEnv, Role
from competition_template.agents_user.random_agent import RandomAgent

# roles: 1 wolf, 1 villager, 1 seer
roles = [Role.WOLF, Role.VILLAGER, Role.SEER]

def main():
    env = WerewolfEnv(roles, talk_history_len=10, max_nights=3)
    agents = {}
    for i, r in enumerate(roles):
        agents[str(i)] = RandomAgent(i, len(roles), r)
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
