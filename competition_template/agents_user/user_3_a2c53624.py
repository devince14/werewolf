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
        """
        初始化智能体
        
        Args:
            agent_id: 自己的编号（0..N-1）
            num_agents: 总玩家数 N
            role: 角色类型
        """
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.role = role
        
        # 在这里添加你的初始化代码
        # 例如：记录游戏状态、策略参数等
        
    def act(self, env):
        """
        根据当前环境状态选择行动
        
        Args:
            env: 狼人杀环境对象
            
        Returns:
            numpy.array: 行动数组
                - talk阶段: [claim_seer, talk_type, target]
                - vote阶段: [target]
                - night阶段: [target]
        """
        # 获取当前阶段
        stage = env.stage
        
        if stage == "talk":
            # 发言阶段
            # claim_seer: 是否声称预言家 (0或1)
            # talk_type: 发言类型 (TalkType枚举)
            # target: 目标玩家编号
            return np.array([0, TalkType.CLAIM_GOOD, env.N])
            
        elif stage == "vote":
            # 投票阶段
            # target: 要投票的玩家编号
            alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
            if alive_players:
                return np.array([alive_players[0]])  # 投第一个存活玩家
            return np.array([env.N])  # 默认不投票
            
        elif stage == "night":
            # 夜晚阶段
            # target: 目标玩家编号
            alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
            
            if self.role == Role.SEER:
                # 预言家查验
                if alive_players:
                    return np.array([alive_players[0]])  # 查验第一个存活玩家
                return np.array([env.N])
            elif self.role == Role.WOLF:
                # 狼人杀人
                if alive_players:
                    return np.array([alive_players[0]])  # 杀第一个存活玩家
                return np.array([env.N])
            else:
                # 村民夜晚无行动
                return np.array([env.N])
        
        return np.array([env.N])
