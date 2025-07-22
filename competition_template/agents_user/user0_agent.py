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
            # 总是投票1号玩家（如果活着且不是自己）
            target = 1 if env.alive[1] and self.agent_id != 1 else env.N
            return np.array([target])
        elif stage == "night":
            if self.role == Role.SEER or self.role == Role.WOLF:
                # 总是查验/杀2号玩家（如果活着且不是自己）
                target = 2 if env.alive[2] and self.agent_id != 2 else env.N
                return np.array([target])
            else:
                return np.array([env.N])
        return np.array([env.N])