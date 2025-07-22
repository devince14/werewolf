"""
狼人杀智能体模板 - 参赛选手需要实现UserAgent类

重要说明：
1. 参赛选手会随机分配到狼人、预言家或村民角色
2. 只需要实现UserAgent类，不需要继承BaseAgent
3. 必须实现__init__和act方法
4. 不要添加额外的辅助函数或属性

游戏配置：1狼人 + 3村民 + 1预言家
评测方式：每种角色都会进行测试，计算总体胜率
"""

import numpy as np
import random
from werewolf_env.werewolf_env import WerewolfEnv, Role, TalkType

class UserAgent:
    """
    狼人杀智能体 - 参赛选手需要实现这个类
    
    要求：
    - 实现__init__方法：初始化智能体
    - 实现act方法：根据当前阶段返回行动
    - 支持三种角色：狼人、预言家、村民
    """
    
    def __init__(self, agent_id: int, num_agents: int, role: Role):
        """
        初始化智能体
        
        Args:
            agent_id: 智能体ID (0-4)
            num_agents: 总智能体数量 (5)
            role: 角色 (Role.WOLF, Role.SEER, Role.VILLAGER)
        """
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.role = role
        
    def act(self, env):
        """
        根据当前阶段选择行动
        
        Args:
            env: 游戏环境对象
            
        Returns:
            numpy.array: 行动数组
        """
        stage = env.stage
        
        if stage == "talk":
            # 发言阶段
            if self.role == Role.SEER:
                # 预言家：声称身份
                return np.array([1, TalkType.CLAIM_SEER, env.N])
            else:
                # 狼人和村民：声称好人身份
                return np.array([0, TalkType.CLAIM_GOOD, env.N])
            
        elif stage == "vote":
            # 投票阶段
            alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
            if alive_players:
                return np.array([random.choice(alive_players)])  # 随机投票
            return np.array([env.N])  # 默认不投票
            
        elif stage == "night":
            # 夜晚阶段
            alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
            
            if self.role == Role.SEER:
                # 预言家查验
                if alive_players:
                    return np.array([random.choice(alive_players)])  # 随机查验
                return np.array([env.N])
            elif self.role == Role.WOLF:
                # 狼人杀人
                if alive_players:
                    return np.array([random.choice(alive_players)])  # 随机杀人
                return np.array([env.N])
            else:
                # 村民夜晚无行动
                return np.array([env.N])
        
        return np.array([env.N]) 