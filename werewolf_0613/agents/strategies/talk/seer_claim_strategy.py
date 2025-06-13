from typing import Tuple
from werewolf_env import WerewolfEnv, TalkType, Role
from ..base.base_strategy import TalkStrategy
import numpy as np

class SeerClaimStrategy(TalkStrategy):
    """预言家声称策略"""
    
    def execute(self, env: WerewolfEnv) -> tuple:
        """返回 (claim_seer, talk_type, target)"""
        # 如果有验人结果，公布结果
        if env.seer_records and env.seer_records[-1][0] == env.day - 1:
            target = env.seer_records[-1][1]
            if env.seer_records[-1][2] == Role.WOLF:
                return 0, TalkType.ACCUSE, target
            else:
                return 0, TalkType.SUPPORT, target
        return 0, TalkType.CLAIM_GOOD, env.N

    def execute_old(self, env: WerewolfEnv) -> Tuple[int, int]:
        # 如果有验人结果，优先公布
        if env.seer_record and env.seer_record[0] == env.day - 1:
            target = env.seer_record[1]
            if env.seer_record[2] == Role.WOLF:
                return TalkType.ACCUSE, target
            else:
                return TalkType.SUPPORT, target
                
        # 没有验人结果时，基于信念选择怀疑对象
        cands = [i for i in env.alive if i != self.agent.agent_id]
        if not cands:
            return TalkType.CLAIM_GOOD, env.N
            
        ps = np.array([self.agent.belief.P_wolf[i] for i in cands])
        if ps.sum() > 0:
            ps /= ps.sum()
            target = int(np.random.choice(cands, p=ps))
            return TalkType.ACCUSE, target
            
        return TalkType.CLAIM_GOOD, env.N 