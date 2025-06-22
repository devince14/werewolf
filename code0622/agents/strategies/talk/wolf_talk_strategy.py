from werewolf_env import WerewolfEnv, TalkType, Role
from ..base.base_strategy import TalkStrategy
import random
import numpy as np

class WolfTalkStrategy(TalkStrategy):
    """狼人的发言策略"""
    
    def __init__(self, agent):
        super().__init__(agent)
    
    def execute(self, env: WerewolfEnv) -> tuple:
        """
        返回 (claim_seer, talk_type, target)
        """
        my = self.agent.agent_id
        alive = [i for i, a in enumerate(env.alive) if a]

        # 决定是否声称预言家
        claim_seer = 0
        # 检查是否有其他狼人已经跳过预言家
        wolf_teammates = self.agent._get_wolf_teammates(env)
        # 将生成器表达式改为列表推导式，并将wolf_teammates作为外部变量
        other_wolf_claimed = False
        for log in env.public_log:
            if isinstance(log, tuple):
                if log[2] == TalkType.CLAIM_SEER and log[1] in wolf_teammates:
                    other_wolf_claimed = True
                    break
            elif isinstance(log, dict):
                if log.get("type") == TalkType.CLAIM_SEER and log.get("speaker") in wolf_teammates:
                    other_wolf_claimed = True
                    break
        
        if (not self.agent.has_claimed_seer and 
            not other_wolf_claimed):
            # 在合适时机跳预言家
            if random.random() < 0.3:  # 50%概率跳预言家
                claim_seer = 1
                self.agent.has_claimed_seer = True

        # 如果是假预言家，随机选择指控好人或支持他人
        if self.agent.has_claimed_seer:
            if random.random() < 0.5:  # 50%概率指控
                # 随机选择一个好人指控
                good_players = [i for i in alive if i != my and i not in wolf_teammates]
                if good_players:
                    return claim_seer, TalkType.ACCUSE, random.choice(good_players)
            # 50%概率支持
            others = [i for i in alive if i != my]
            if others:
                return claim_seer, TalkType.SUPPORT, random.choice(others)
            return claim_seer, TalkType.CLAIM_GOOD, env.N

        # 如果不是假预言家，就装好人
        return claim_seer, TalkType.CLAIM_GOOD, env.N 