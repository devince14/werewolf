from werewolf_env import Role
from agents.strategies.talk.seer_talk_strategy import SeerTalkStrategy
from agents.strategies.talk.wolf_talk_strategy import WolfTalkStrategy
from agents.strategies.talk.villager_talk_strategy import VillagerTalkStrategy
from agents.strategies.vote.belief_based_vote import BeliefBasedVoteStrategy
from agents.strategies.night.wolf_kill_strategy import WolfKillStrategy
from agents.strategies.night.seer_check_strategy import SeerCheckStrategy
from .belief.role_specific_belief_update import RoleSpecificBeliefUpdate

class StrategyFactory:
    """策略工厂，用于创建和管理不同角色的策略组合"""
    
    @staticmethod
    def create_strategies(agent, role: Role):
        """
        根据角色创建策略组合
        
        Args:
            agent: Agent实例
            role: 角色类型
            
        Returns:
            dict: 包含各个阶段策略的字典
        """
        strategies = {
            'belief_update': RoleSpecificBeliefUpdate(agent)
        }
        
        if role == Role.SEER:
            strategies.update({
                'talk': SeerTalkStrategy(agent),
                'vote': BeliefBasedVoteStrategy(agent, threshold=0.6),
                'night': SeerCheckStrategy(agent)
            })
        elif role == Role.WOLF:
            strategies.update({
                'talk': WolfTalkStrategy(agent),
                'vote': BeliefBasedVoteStrategy(agent, threshold=0.4),
                'night': WolfKillStrategy(agent)
            })
        else:  # 村民
            strategies.update({
                'talk': VillagerTalkStrategy(agent),
                'vote': BeliefBasedVoteStrategy(agent, threshold=0.5),
                'night': None
            })
            
        return strategies 
