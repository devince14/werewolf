import numpy as np
from werewolf_env.werewolf_env import WerewolfEnv, Role, TalkType

class UserAgent:
    """
    你的狼人杀智能体代码
    请实现以下方法：
    - __init__: 初始化
    - act: 主要行动方法
    
    注意：这个版本不继承BaseAgent，直接实现所需接口
    """
    
    def __init__(self, agent_id: int, num_agents: int, role: Role):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.role = role
        self.has_claimed_seer = False  # 记录是否声称过预言家

    def act(self, env: WerewolfEnv):
        if env.stage == "talk":
            # 不跳预言家，只声称好人
            return np.array([0, TalkType.CLAIM_GOOD, env.N])
        if env.stage == "vote":
            if self.role == Role.WOLF:
                # 狼人随机投票给一个活着的非狼玩家
                alive_non_wolves = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
                if alive_non_wolves:
                    return np.array([np.random.choice(alive_non_wolves)])
            return np.array([env.N])  # 其他角色返回无效动作
        if env.stage == "night":
            if self.role == Role.WOLF:
                # 狼人随机杀死一个活着的非狼玩家
                alive_non_wolves = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
                if alive_non_wolves:
                    return np.array([np.random.choice(alive_non_wolves)])
            elif self.role == Role.SEER:
                return np.array([0])  # 验证0号玩家
        return np.array([env.N])
