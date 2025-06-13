from werewolf_env import WerewolfEnv, Role, TalkType
from ..base.base_strategy import VoteStrategy
import numpy as np

class BeliefBasedVoteStrategy(VoteStrategy):
    """基于信念的投票策略"""
    
    def __init__(self, agent, threshold=0.5):
        super().__init__(agent)
        self.threshold = threshold
    
    def execute(self, env: WerewolfEnv) -> int:
        """返回投票目标"""
        my = self.agent.agent_id
        alive = [i for i, a in enumerate(env.alive) if a]
        if not alive or my not in alive:
            return env.N

        # 如果是预言家且有验出狼，优先投票
        if (self.agent.role == Role.SEER and 
            env.seer_records and 
            env.seer_records[-1][2] == Role.WOLF):
            target = env.seer_records[-1][1]
            if target in alive:
                return target

        # 狼人的投票策略
        if self.agent.role == Role.WOLF:
            wolf_teammates = self.agent._get_wolf_teammates(env)
            
            # 检查是否有狼人跳预言家并指控了某人
            for event in env.event_log:
                if event.get("phase") == "talk":
                    speaker = event.get("speaker")
                    if ((speaker == my or speaker in wolf_teammates) and 
                        env.alive[speaker] and 
                        event.get("type") == TalkType.CLAIM_SEER):
                        # 找到这个狼人后续的指控
                        for later_event in env.event_log:
                            if (later_event.get("phase") == "talk" and
                                later_event.get("speaker") == speaker and
                                later_event.get("type") == TalkType.ACCUSE):
                                target = later_event.get("target")
                                if target in alive:
                                    return target
            
            # 如果没有狼人跳预言家，则按预言家信念投票
            cands = [i for i in alive if i not in wolf_teammates and i != my]
            if not cands:
                return env.N
            
            # 选择最可能是预言家的目标
            p_seer = np.array([self.agent.belief.P_seer[i] for i in cands])
            if p_seer.sum() > 0:
                p_seer = p_seer / p_seer.sum()
                return int(cands[np.argmax(p_seer)])
            
            # 如果没有明显目标，基于概率随机选择
            uniform_probs = np.ones(len(cands)) / len(cands)
            return int(np.random.choice(cands, p=uniform_probs))

        # 好人阵营（预言家和村民）的投票策略
        cands = [i for i in alive if i != my]
        if not cands:
            return env.N
        
        # 按狼人概率投票
        p_wolf = np.array([self.agent.belief.P_wolf[i] for i in cands])
        if p_wolf.sum() > 0:
            p_wolf = p_wolf / p_wolf.sum()
            return int(cands[np.argmax(p_wolf)])
        
        # 如果没有明显目标，基于概率随机选择，而不是完全随机
        # 这样可以避免总是选择第一个候选人
        uniform_probs = np.ones(len(cands)) / len(cands)

        # ------ DEBUG ------
        import logging
        log = logging.getLogger(__name__)

        if self.agent.role != Role.WOLF:          # 好人阵营
            info = ", ".join([f"{i}:{self.agent.belief.P_wolf[i]:.2f}" for i in cands])
            log.debug("Agent %s vote.  P_wolf: [%s] -> pick %s",
                    my, info, target if 'target' in locals() else 'skip')
        else:                                     # 狼阵营
            info = ", ".join([f"{i}:{self.agent.belief.P_seer[i]:.2f}" for i in cands])
            log.debug("Wolf %s vote.  P_seer: [%s] -> pick %s",
                    my, info, target if 'target' in locals() else 'skip')
        # -------------------


        return int(np.random.choice(cands, p=uniform_probs))

        
            

        