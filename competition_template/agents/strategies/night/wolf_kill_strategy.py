from werewolf_env import WerewolfEnv, Role
from ..base.base_strategy import NightStrategy
import numpy as np
import random

class WolfKillStrategy(NightStrategy):
    """狼人刀人策略"""
    
    def execute(self, env: WerewolfEnv) -> int:
        """选择一个目标击杀"""
        try:
            # 获取自己的ID和存活玩家列表
            my = self.agent.agent_id
            
            # 确保 env.alive 是有效的
            if env.alive is None:
                return env.N
                
            alive = [i for i, a in enumerate(env.alive) if a]
            
            # 获取狼队友
            wolf_teammates = []
            try:
                if hasattr(self.agent, '_get_wolf_teammates'):
                    wolf_teammates = self.agent._get_wolf_teammates(env) or []
            except Exception as e:
                wolf_teammates = []
            
            # 获取候选目标（排除自己和队友）
            cands = [i for i in alive if i != my and i not in wolf_teammates]
            if len(cands) == 0:
                return env.N  # 不杀人
            
            # 获取优先目标（声称是预言家的玩家）
            priority_targets = []
            try:
                if (hasattr(self.agent, 'belief') and 
                    hasattr(self.agent.belief, 'claimed_seers')):
                    priority_targets = [
                        p for p in cands
                        if p in self.agent.belief.claimed_seers
                    ]
            except Exception:
                priority_targets = []
            
            # 处理优先目标
            if priority_targets:
                try:
                    if (hasattr(self.agent, 'belief') and 
                        hasattr(self.agent.belief, 'P_seer')):
                        # 获取每个目标的预言家概率
                        p_seer_values = []
                        for x in priority_targets:
                            try:
                                p_seer_values.append(float(self.agent.belief.P_seer[x]))
                            except (IndexError, TypeError, ValueError):
                                p_seer_values.append(0.0)
                        
                        if p_seer_values:  # 确保有有效的概率值
                            # 将目标和概率打包在一起排序
                            sorted_targets = sorted(zip(priority_targets, p_seer_values), 
                                                 key=lambda x: x[1], 
                                                 reverse=True)
                            return int(sorted_targets[0][0])
                except Exception:
                    pass
            
            # 如果没有优先目标或处理失败，使用预言家概率选择目标
            try:
                if (hasattr(self.agent, 'belief') and 
                    hasattr(self.agent.belief, 'P_seer')):
                    # 获取每个候选人的预言家概率
                    P_vals = []
                    for i in cands:
                        try:
                            P_vals.append(float(self.agent.belief.P_seer[i]))
                        except (IndexError, TypeError, ValueError):
                            P_vals.append(0.0)
                    
                    P_vals = np.array(P_vals, dtype=np.float64)
                    
                    # 如果所有概率都是0，随机选择
                    if P_vals.sum() == 0:
                        return int(random.choice(cands))
                    
                    # 归一化概率
                    ps = P_vals / P_vals.sum()
                    
                    # 使用 random.choices 选择目标
                    return int(random.choices(cands, weights=ps, k=1)[0])
            except Exception:
                pass
            
            # 如果概率计算失败，随机选择
            return int(random.choice(cands)) if cands else env.N
            
        except Exception:
            return env.N 
