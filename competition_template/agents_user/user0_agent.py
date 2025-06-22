import numpy as np
from competition_template.werewolf_env.werewolf_env import WerewolfEnv, Role, TalkType

class User0Agent:
    """Example contestant agent with simple deterministic behavior."""

    def __init__(self, agent_id: int, num_agents: int, role: Role):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.role = role

    def act(self, env: WerewolfEnv):
        if env.stage == "talk":
            # Always claim to be good with no target
            return np.array([TalkType.CLAIM_GOOD, 0, env.N])
        if env.stage == "vote":
            # Vote for the lowest-numbered alive player other than self
            for i in range(self.num_agents):
                if i != self.agent_id and env.alive[i]:
                    return i
            return env.N
        if env.stage == "night":
            if self.role == Role.WOLF:
                # Kill the lowest-numbered alive non-wolf
                for i in range(self.num_agents):
                    if env.alive[i] and i != self.agent_id:
                        return i
                return env.N
            if self.role == Role.SEER:
                # Check the lowest-numbered alive player other than self
                for i in range(self.num_agents):
                    if i != self.agent_id and env.alive[i]:
                        return i
                return env.N
            return 0
        return 0
