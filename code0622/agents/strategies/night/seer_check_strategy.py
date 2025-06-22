from werewolf_env import WerewolfEnv, TalkType, Role
from ..base.base_strategy import NightStrategy
import random
import numpy as np

class SeerCheckStrategy(NightStrategy):
    """预言家的夜间查验策略"""
    
    def execute(self, env: WerewolfEnv) -> int:
        """选择一个目标进行查验"""
        try:
            my = self.agent.agent_id
            alive = [i for i, a in enumerate(env.alive) if a]
            
            # 获取候选玩家（存活、非自己、未查验过）
            cands = [i for i in alive
                    if i != my and i not in self.agent.checked_players]

            if not cands:  # 所有玩家都已查验过
                return env.N

            # 计算每个候选玩家的查验优先级分数
            scores = []
            for player in cands:
                try:
                    score = self.agent.belief.P_wolf[player]  # 基础分=狼人概率

                    # 调整因素：被支持次数（被支持越多越不可疑）
                    support_count = len(self.agent.belief.supported_by.get(player, []))
                    score *= (0.9 ** support_count)
                    scores.append(score)
                except Exception:
                    scores.append(0.0)

            # 归一化概率
            scores = np.array(scores, dtype=np.float64)
            if scores.sum() > 0:
                scores /= scores.sum()
                target = int(np.random.choice(cands, p=scores))
            else:
                target = int(random.choice(cands))

            # 记录已查验
            self.agent.checked_players.add(target)
            return target
            
        except Exception:
            return env.N 