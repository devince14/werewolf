from werewolf_env import WerewolfEnv, TalkType, Role
from ..base.base_strategy import TalkStrategy
import random
import numpy as np

class VillagerTalkStrategy(TalkStrategy):
    """村民的发言策略"""
    
    def execute(self, env: WerewolfEnv) -> tuple:
        """
        返回 (claim_seer, talk_type, target)
        """
        my = self.agent.agent_id
        alive = [i for i, a in enumerate(env.alive) if a]

        # 根据对预言家的信任度决定支持谁
        most_trusted_seer = None
        max_credibility = 0
        for seer_id, cred in self.agent.belief.claimed_seers.items():
            if seer_id in alive and cred > max_credibility:
                most_trusted_seer = seer_id
                max_credibility = cred

        # 如果有可信的预言家，有机会支持他
        if most_trusted_seer and max_credibility > 0.7:
            if random.random() < 0.4:  # 40%概率支持可信预言家
                return 0, TalkType.SUPPORT, most_trusted_seer

        # 否则按狼人信念发言
        cands = alive.copy()
        # 找到信念最高者及其概率
        P_vals = np.array([self.agent.belief.P_wolf[i] for i in cands])
        idx_max = np.argmax(P_vals)
        best_p = P_vals[idx_max]
        best_target = cands[idx_max]

        if best_p < self.agent.tau_info:
            # 信息不足，自证优先
            return 0, TalkType.CLAIM_GOOD, env.N

        # 如果最高概率超过阈值，直接选择该目标
        if best_target != my:  # 确保不是自己
            return 0, TalkType.ACCUSE, best_target

        return 0, TalkType.CLAIM_GOOD, env.N 
