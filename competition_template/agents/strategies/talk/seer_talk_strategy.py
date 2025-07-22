from werewolf_env import WerewolfEnv, TalkType, Role
from agents.strategies.base.base_strategy import TalkStrategy
import numpy as np

class SeerTalkStrategy(TalkStrategy):
    """预言家的发言策略"""
    
    def execute(self, env: WerewolfEnv) -> tuple:
        """
        返回 (claim_seer, talk_type, target)
        """
        my = self.agent.agent_id
        alive = [i for i, a in enumerate(env.alive) if a]
        # 决定是否声称预言家
        claim_seer = 0
        if not self.agent.has_claimed_seer:
            # 第一天特殊处理
            if env.day == 1:  # 第一天是day=1
                # 检查是否有人跳预言家
                has_other_seer = any(getattr(agent, 'has_claimed_seer', False) for i, agent in enumerate(env.agents) if i != my)
                # 检查第一晚验人结果
                found_wolf = (env.seer_records and 
                            len(env.seer_records) > 0 and 
                            env.seer_records[0][0] == 0 and 
                            env.seer_records[0][2] == Role.WOLF)
                # 如果没人跳预言家且第一晚没验出狼，且自己还活着，则不跳预言家
                if not has_other_seer and not found_wolf and env.alive[my]:
                    return claim_seer, TalkType.CLAIM_GOOD, env.N
            
            # 第一天验出狼人，或第二天及以后，或有其他人跳预言家时，直接跳预言家
            claim_seer = 1
            self.agent.has_claimed_seer = True
            
            # 第二天跳预言家时，策略只需要返回声明，具体的验人结果由belief_agent处理
            if env.day == 2:
                return claim_seer, TalkType.CLAIM_GOOD, env.N
            
            # 其他情况：如果有验人结果，返回最新的结果
            elif env.seer_records:
                day, target, role = env.seer_records[-1]
                if day == env.day - 1:  # 只报告昨晚的结果
                    if role == Role.WOLF:
                        return claim_seer, TalkType.ACCUSE, target
                    else:
                        return claim_seer, TalkType.SUPPORT, target
            
            return claim_seer, TalkType.CLAIM_GOOD, env.N

        # 如果已经声称过预言家，且有新的验人结果要公布
        if self.agent.has_claimed_seer and env.seer_records and len(env.seer_records) > 0:
            latest_record = env.seer_records[-1]
            if latest_record[0] == env.day - 1:  # 昨晚的验人结果
                target = latest_record[1]
                if latest_record[2] == Role.WOLF:
                    return 0, TalkType.ACCUSE, target
                else:
                    return 0, TalkType.SUPPORT, target

        # 如果没有新的验人结果，根据信念随机指控
        if self.agent.has_claimed_seer:
            cands = [i for i in alive if i != my]
            if cands:
                ps = np.array([self.agent.belief.P_wolf[i] for i in cands])
                if ps.sum() > 0:
                    ps /= ps.sum()
                    target = int(np.random.choice(cands, p=ps))
                    return 0, TalkType.ACCUSE, target

        return 0, TalkType.CLAIM_GOOD, env.N 
