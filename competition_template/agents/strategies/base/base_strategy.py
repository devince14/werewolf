from abc import ABC, abstractmethod
from typing import Any, Tuple
from competition_template.werewolf_env import WerewolfEnv

class Strategy(ABC):
    """策略接口基类"""
    
    def __init__(self, agent):
        self.agent = agent
        
    @abstractmethod
    def execute(self, env: WerewolfEnv) -> Any:
        """执行策略"""
        pass
        
class TalkStrategy(Strategy):
    """发言策略基类"""
    
    @abstractmethod
    def execute(self, env: WerewolfEnv) -> Tuple[int, int]:
        """返回(talk_type, target)"""
        pass

class VoteStrategy(Strategy):
    """投票策略基类"""
    
    @abstractmethod
    def execute(self, env: WerewolfEnv) -> int:
        """返回投票目标"""
        pass

class NightStrategy(Strategy):
    """夜晚策略基类"""
    
    @abstractmethod
    def execute(self, env: WerewolfEnv) -> int:
        """返回夜晚行动目标"""
        pass

class BeliefUpdateStrategy(Strategy):
    """信念更新策略基类"""
    
    @abstractmethod
    def execute(self, env: WerewolfEnv, events: list) -> None:
        """更新信念"""
        pass 
