import numpy as np
import random
from werewolf_env.werewolf_env import Role, TalkType

class UserAgent:
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
            alive = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
            return np.array([random.choice(alive)]) if alive else np.array([env.N])
        elif stage == "night":
            alive = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
            if self.role == Role.SEER or self.role == Role.WOLF:
                return np.array([random.choice(alive)]) if alive else np.array([env.N])
            else:
                return np.array([env.N])
        return np.array([env.N])