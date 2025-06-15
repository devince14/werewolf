from abc import ABC
from typing import Any, Dict
import numpy as np
from werewolf_env import WerewolfEnv, Role
from agents.strategies import StrategyFactory

class BaseAgent(ABC):
    """
    Agent基类。所有具体Agent都继承自此类。
    使用策略模式处理不同阶段的行为。
    """

    def __init__(self, agent_id: int, num_agents: int, role: Role):
        """
        初始化Agent
        
        Args:
            agent_id: 自己的编号（0..N-1）
            num_agents: 总玩家数 N
            role: 角色类型
        """
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.role = role
        
        # 用于记录上次读到public_log的位置
        self._last_event_idx = 0
        
        # 初始化信念状态
        self.belief = self._init_belief()
        
        # 加载策略
        self.strategies = StrategyFactory.create_strategies(self, role)
        
    def _init_belief(self) -> Dict:
        """初始化信念状态"""
        return {
            'P_wolf': np.ones(self.num_agents) / self.num_agents,  # 是狼人的概率
            'P_seer': np.ones(self.num_agents) / self.num_agents,  # 是预言家的概率
            'claimed_seers': {},  # 声称是预言家的玩家
            'supported_by': {},   # 被谁支持
        }
        
    def fetch_new_events(self, env: WerewolfEnv) -> list:
        """获取新事件并更新信念"""
        all_events = env.public_log
        new_events = all_events[self._last_event_idx:]
        self._last_event_idx = len(all_events)
        
        # 更新信念
        if new_events and self.strategies['belief_update']:
            self.strategies['belief_update'].execute(env, new_events)
            
        return new_events  # 返回新事件列表

    def act(self, env: WerewolfEnv) -> Any:
        """根据当前阶段执行相应的策略"""
        # 获取并处理新事件
        self.fetch_new_events(env)
        
        # 根据当前阶段选择策略
        if env.phase == 'TALK':
            strategy = self.strategies.get('talk')
        elif env.phase == 'VOTE':
            strategy = self.strategies.get('vote')
        elif env.phase == 'NIGHT':
            strategy = self.strategies.get('night')
        else:
            return env.N
            
        # 执行策略
        if strategy:
            return strategy.execute(env)
        return env.N