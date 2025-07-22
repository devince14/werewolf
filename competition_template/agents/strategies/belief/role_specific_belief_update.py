from werewolf_env import WerewolfEnv, TalkType, Role
from agents.strategies.base.base_strategy import BeliefUpdateStrategy, VoteStrategy
import numpy as np

class RoleSpecificBeliefUpdate(BeliefUpdateStrategy):
    """基于角色的信念更新策略"""
    
    def __init__(self, agent=None):
        super().__init__(agent)
        # 贝叶斯更新参数
        self.alpha_wolf = 0.1  # 好人误指控村民的概率
        self.beta_wolf = 0.9   # 好人指控狼人的概率
        self.alpha_seer = 0.05  # 预言家误指控的概率（比普通人更低）
        self.beta_seer = 0.95   # 预言家正确指控的概率（比普通人更高）
        
        # 各种行为的影响系数
        self.fake_seer_penalty = 0.8    # 假预言家的惩罚系数
        self.normal_accuse_penalty = 0.4 # 普通指控的惩罚系数
        self.trust_gain = 0.3           # 支持带来的信任增益
        self.seer_check_confidence = 0.95 # 预言家验人的可信度（应该是最高的）
        self.wolf_teammate_suspicion = 0.4 # 对狼人支持对象的怀疑系数
        self.seer_death_penalty = 0.7    # 当一个预言家死亡时，对另一个预言家的狼人概率增加
        
        # 添加新的跟踪变量
        self.player_behavior_history = {}  # 记录每个玩家的行为历史
        self.contradiction_penalty = 0.7   # 矛盾行为惩罚系数
        
    def execute(self, env: WerewolfEnv, events: list) -> None:
        # 首先检查新事件中是否有死亡信息
        for event in events:
            if isinstance(event, tuple) and len(event) >= 5:
                phase, speaker, talk_type, target, role_int = event
                if phase == 'night' and target >= 0:  # 有人死亡
                    # 检查死者是否是预言家
                    claimed_seers = getattr(self.agent.belief, 'claimed_seers', {})
                    if target in claimed_seers:
                        self._handle_seer_death(env)
                        break  # 找到死亡的预言家就可以停止检查

        # 然后处理其他事件
        for event in events:
            if isinstance(event, tuple) and len(event) >= 5:
                phase, speaker, talk_type, target, role_int = event
                
                # 忽略自证好人的发言
                if talk_type == TalkType.CLAIM_GOOD:
                    continue
                    
                # 更新预言家声称
                if talk_type == TalkType.CLAIM_SEER:
                    self._handle_seer_claim(env, speaker)

                # 更新支持关系（包括验出好人的结果）
                elif talk_type == TalkType.SUPPORT:
                    claimed_seers = getattr(self.agent.belief, 'claimed_seers', {})
                    if speaker in claimed_seers and role_int == Role.VILLAGER:
                        # 预言家验出好人的情况
                        credibility = claimed_seers[speaker]
                        self.agent.belief.P_wolf[target] = max(0.0,
                            self.agent.belief.P_wolf[target] - self.seer_check_confidence * credibility)
                        # print(f"[DEBUG] Seer {speaker} checked {target} as VILLAGER, updating belief to {self.agent.belief.P_wolf[target]}")
                    else:
                        # 普通的支持行为
                        self._handle_support(env, speaker, target)

                # 更新指控关系（包括验出狼人的结果）
                elif talk_type == TalkType.ACCUSE:
                    self._handle_accuse(env, speaker, target, role_int)
                        
        # 归一化概率
        #self._normalize_probabilities(env)
    
    def _handle_seer_death(self, env: WerewolfEnv):
        """处理预言家死亡的情况"""
        # 获取所有声称是预言家的玩家
        claimed_seers = getattr(self.agent.belief, 'claimed_seers', {})
        claimed_seers_list = list(claimed_seers.keys())
        if len(claimed_seers_list) >= 2:  # 如果有两个或以上的预言家
            # 检查是否有预言家在夜里死亡
            for seer in claimed_seers_list:
                if not env.alive[seer]:  # 如果这个预言家死了
                    # 增加其他声称预言家的狼人概率
                    for other_seer in claimed_seers_list:
                        if env.alive[other_seer] and other_seer != seer:  # 对其他还活着的预言家
                            old_prob = self.agent.belief.P_wolf[other_seer]
                            # 大幅增加狼人概率
                            self.agent.belief.P_wolf[other_seer] = min(1.0,
                                old_prob + self.seer_death_penalty)
                            # 同时降低其预言家概率
                            if hasattr(self.agent.belief, 'P_seer'):
                                self.agent.belief.P_seer[other_seer] = max(0.0,
                                    self.agent.belief.P_seer[other_seer] - self.seer_death_penalty)
                    break  # 找到一个死亡的预言家就足够了

    def _handle_seer_claim(self, env: WerewolfEnv, speaker: int):
        """处理预言家声称"""
        claimed_seers = getattr(self.agent.belief, 'claimed_seers', {})
        claimed_seers[speaker] = 0.5
        
        # 如果我是真预言家，且有人冒充，极大提高对其狼人嫌疑
        if env.roles[self.agent.agent_id] == Role.SEER and speaker != self.agent.agent_id:
            self.agent.belief.P_wolf[speaker] = min(1.0, self.agent.belief.P_wolf[speaker] + self.fake_seer_penalty)
            claimed_seers[speaker] = 0.0
            
        # 如果我是狼人，知道队友的真实身份
        elif env.roles[self.agent.agent_id] == Role.WOLF:
            if speaker in self.agent._get_wolf_teammates(env):
                claimed_seers[speaker] = 0.0
                
    def _check_contradiction(self, env: WerewolfEnv, speaker: int, target: int, action_type: TalkType):
        """检查行为是否与历史矛盾"""
        if speaker not in self.player_behavior_history:
            self.player_behavior_history[speaker] = []
        
        history = self.player_behavior_history[speaker]
        contradiction = False
        
        # 检查是否与历史行为矛盾
        for past_action in reversed(history):  # 从最近的历史开始检查
            # 计算时间衰减因子
            days_passed = env.day - past_action.get('day', 0)
            if days_passed > 2:  # 超过2天的历史行为影响较小
                continue
                
            #decay_factor = 1.0 / (1.0 + days_passed)  # 时间衰减因子
            decay_factor = 1.0   # 时间衰减因子
            
            # 检查矛盾行为
            if (action_type == TalkType.SUPPORT and past_action.get('type') == TalkType.ACCUSE and 
                target == past_action.get('target')):
                contradiction = True
                self.contradiction_penalty *= decay_factor
                break
            elif (action_type == TalkType.ACCUSE and past_action.get('type') == TalkType.SUPPORT and 
                  target == past_action.get('target')):
                contradiction = True
                self.contradiction_penalty *= decay_factor
                break
                
        # 记录新行为
        history.append({
            'type': action_type,
            'target': target,
            'day': env.day
        })
        
        return contradiction
        
    def _handle_support(self, env: WerewolfEnv, supporter_id: int, target_id: int):
        """处理支持行为"""
        # 检查矛盾
        if self._check_contradiction(env, supporter_id, target_id, TalkType.SUPPORT):
            # 增加说话者的狼人概率
            self.agent.belief.P_wolf[supporter_id] = min(1.0,
                self.agent.belief.P_wolf[supporter_id] + self.contradiction_penalty)
        
        if target_id not in self.agent.belief.supported_by:
            self.agent.belief.supported_by[target_id] = set()
        self.agent.belief.supported_by[target_id].add(supporter_id)
        
        # 如果支持者是预言家，增加被支持者的可信度
        claimed_seers = getattr(self.agent.belief, 'claimed_seers', {})
        if supporter_id in claimed_seers:
            credibility = claimed_seers[supporter_id]
            self.agent.belief.P_wolf[target_id] = max(0.0, 
                self.agent.belief.P_wolf[target_id] - self.trust_gain * credibility)
            
        # 如果被支持者是我自己
        if target_id == self.agent.agent_id:
            if supporter_id in claimed_seers:
                # 如果支持我的人是预言家，增加其可信度
                claimed_seers[supporter_id] = min(1.0,
                    claimed_seers[supporter_id] + self.trust_gain)
                    
    def _handle_accuse(self, env: WerewolfEnv, speaker: int, target: int, role_int: int):
        """处理指控和预言家验人结果"""
        # 检查矛盾
        if self._check_contradiction(env, speaker, target, TalkType.ACCUSE):
            # 增加说话者的狼人概率
            self.agent.belief.P_wolf[speaker] = min(1.0,
                self.agent.belief.P_wolf[speaker] + self.contradiction_penalty)
                
        # 预言家指控（包括验出狼人结果）
        # print(self.agent.belief)
        claimed_seers = getattr(self.agent.belief, 'claimed_seers', {})
        if speaker in claimed_seers:
            
            credibility = claimed_seers[speaker]
            # 如果指控目标是我自己
            if target == self.agent.agent_id and env.roles[self.agent.agent_id] != Role.WOLF:
                self._handle_accuse_me(speaker, speaker in claimed_seers)
                claimed_seers[speaker] = 0.0
            # 如果我是真预言家，且说话的不是我
            elif env.roles[self.agent.agent_id] == Role.SEER and speaker != self.agent.agent_id:
                # 降低这个预言家的可信度，验人结果的惩罚更重
                penalty = self.fake_seer_penalty if role_int == Role.WOLF else self.normal_accuse_penalty
                claimed_seers[speaker] = 0.0
                # 不更新对目标的狼人概率
            else:
                # 其他人正常更新信念
                # 如果是验人结果（role_int不为-1）
                if role_int != -1:
                    # 根据预言家的可信度更新
                    if role_int == Role.WOLF:  # 验出是狼人
                        self.agent.belief.P_wolf[target] = min(1.0,
                            self.agent.belief.P_wolf[target] + self.seer_check_confidence * credibility)
                        # print(f"[DEBUG] Seer {speaker} checked {target} as WOLF, updating belief to {self.agent.belief.P_wolf[target]}")
                    else:  # 验出是好人
                        self.agent.belief.P_wolf[target] = max(0.0,
                            self.agent.belief.P_wolf[target] - self.seer_check_confidence * credibility)
                        # print(f"[DEBUG] Seer {speaker} checked {target} as VILLAGER, updating belief to {self.agent.belief.P_wolf[target]}")
                else:
                    # 普通指控
                    self._update_on_wolf_accuse(speaker, target)
                
        else:
            # 如果指控目标是我自己
            if target == self.agent.agent_id and env.roles[self.agent.agent_id] != Role.WOLF:
                self._handle_accuse_me(speaker, speaker in claimed_seers)
            else:
                # 普通玩家的指控
                self._update_on_wolf_accuse(speaker, target)
            
        
    
    def _update_on_seer_accuse(self, speaker: int, target: int):
        """处理预言家的指控"""
        prior = self.agent.belief.P_wolf[target]
        post_true = prior * self.beta_seer
        post_false = (1-prior) * self.alpha_seer
        norm = post_true + post_false
        if norm > 0:
            self.agent.belief.P_wolf[target] = post_true / norm
            
    def _update_on_wolf_accuse(self, speaker: int, target: int):
        """处理普通玩家的指控"""
        if speaker == self.agent.agent_id or target == self.agent.agent_id:
            return
        prior = self.agent.belief.P_wolf[target]
        post_true = prior * self.beta_wolf
        post_false = (1-prior) * self.alpha_wolf
        norm = post_true + post_false
        if norm > 0:
            self.agent.belief.P_wolf[target] = post_true / norm
            
    def _handle_accuse_me(self, speaker: int, is_claimed_seer: bool):
        """处理被指控的情况"""
        if is_claimed_seer:
            # 自称预言家指控我
            self.agent.belief.P_wolf[speaker] = min(1.0, 
                self.agent.belief.P_wolf[speaker] + self.fake_seer_penalty)
        else:
            # 普通玩家指控我
            self.agent.belief.P_wolf[speaker] = min(1.0,
                self.agent.belief.P_wolf[speaker] + self.normal_accuse_penalty)
    
    def _normalize_probabilities(self, env: WerewolfEnv):
        """归一化概率"""
        total_p = sum(self.agent.belief.P_wolf)
        if total_p > 0:
            for i in range(env.N):
                self.agent.belief.P_wolf[i] /= total_p 

class BeliefBasedVoteStrategy(VoteStrategy):
    def execute(self, env: WerewolfEnv) -> int:
        my = self.agent.agent_id
        alive = [i for i, a in enumerate(env.alive) if a]
        
        # 分析投票历史和发言的一致性
        vote_consistency = {}
        for player in alive:
            if player == my:
                continue
                
            # 检查投票与发言的一致性
            consistency_score = 1.0
            player_votes = []
            player_accusations = []
            
            for event in env.event_log:
                if event.get("speaker") == player:
                    if event.get("type") == TalkType.ACCUSE:
                        player_accusations.append(event.get("target"))
                elif event.get("phase") == "vote" and event.get("voter") == player:
                    player_votes.append(event.get("target"))
            
            # 检查最近的投票是否与发言一致
            if player_votes and player_accusations:
                last_vote = player_votes[-1]
                if last_vote not in player_accusations:
                    consistency_score *= 0.7  # 降低一致性分数
                    
            vote_consistency[player] = consistency_score
        
        # 结合一致性分数和狼人概率
        final_scores = {}
        for player in alive:
            if player == my:
                continue
            
            score = self.agent.belief.P_wolf[player]
            if player in vote_consistency:
                score *= vote_consistency[player]
            final_scores[player] = score
            
        # 选择得分最高的玩家
        if final_scores:
            target = max(final_scores.items(), key=lambda x: x[1])[0]
            return target
            
        return env.N
