import numpy as np
from competition_template.werewolf_env.werewolf_env import WerewolfEnv, Role

class RandomAgent:
    """A baseline agent that takes random legal actions."""
    def __init__(self, agent_id: int, num_agents: int, role: Role):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.role = role
        self.has_claimed_seer = False  # 新增此行

    def act(self, env: WerewolfEnv):
        if env.stage == "talk":
            return env.talk_space.sample()
        if env.stage == "vote":
            return env.vote_space.sample()
        if env.stage == "night":
            if self.role == Role.WOLF:
                return env.kill_space.sample()
            if self.role == Role.SEER:
                return env.seer_space.sample()
            return env.noop_space.sample()
        return np.array(0)
