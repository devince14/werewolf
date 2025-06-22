import numpy as np
from competition_template.werewolf_env import Role

class BeliefState:
    """
    维护两套后验概率：
      - P_wolf[i]: i 是狼人的概率
      - P_seer[i]: i 是预言家的概率
    """

    def __init__(self, num_players: int, self_id: int, role=Role, num_wolves: int = 2):
        self.N = num_players
        self.self_id = self_id
        self.role = role  # 存储玩家角色（类型为Role枚举）
        # 初始化：自己绝对不是狼；其他人按狼人数量设置初始概率
        self.P_wolf = np.ones(self.N, dtype=float) * num_wolves / (self.N - 1)
        self.P_wolf[self_id] = 0.0

        if self.role == Role.WOLF:
            self.P_seer = np.ones(self.N, dtype=float) / (self.N - num_wolves)
            self.P_seer[self_id] = 0.0
        else:
            self.P_seer = np.ones(self.N, dtype=float) / (self.N - 1)
            self.P_seer[self_id] = 0.0

        # 贝叶斯更新参数
        self.alpha_wolf = 0.1  # 好人误 ACCUSE 村民的概率
        self.beta_wolf  = 0.9  # 好人 ACCUSE 狼人的概率

        self.alpha_seer = 0.1  # 非预言家发 ACCUSE 的概率
        self.beta_seer  = 0.9  # 预言家发 ACCUSE 的概率

        self.fake_seer_penalty = 0.8  # 假预言家的惩罚系数
        self.normal_accuse_penalty = 0.4  # 普通指控的惩罚系数

        self.eta_support_truth = 0.5  # 支持真村民的信任增益
        self.delta_expose_teammate = 0.3  # 狼队友暴露后的嫌疑增量
        self.gamma_link = 0.2  # 假预言家暴露后的连带效应系数

        self.trust_gain = 0.3  # 支持带来的信任增益
        self.seer_death_suspicion = 0.6  # 真预言家死亡时对假预言家的怀疑增加
        self.wolf_teammate_suspicion = 0.4  # 对狼人支持对象的怀疑系数

        # 记录声称是预言家的玩家及其可信度
        self.claimed_seers = {}  # {player_id: credibility}
        self.supported_by = {}  # 记录被谁支持过: {player_id: [supporter_ids]}

    def update_on_seer_accuse(self, speaker: int):
        """
        看到一次 accuse 事件，就按贝叶斯更新 P_seer[speaker]：
        P(seer|accuse) ∝ P_seer * beta_seer,  P(~seer|accuse) ∝ (1-P_seer)*alpha_seer
        """
        prior = self.P_seer[speaker]
        post_true  = prior * self.beta_seer
        post_false = (1-prior) * self.alpha_seer
        norm = post_true + post_false
        if norm>0:
            self.P_seer[speaker] = post_true / norm

    def update_on_wolf_accuse(self, speaker: int, target: int):
        """
        普通指控下，用贝叶斯更新 P_wolf[target]。
        """
        # 不更新自己
        if speaker == self.self_id or target == self.self_id:
            return
        prior = self.P_wolf[target]
        post_true  = prior * self.beta_wolf
        post_false = (1-prior) * self.alpha_wolf
        norm = post_true + post_false
        if norm>0:
            self.P_wolf[target] = post_true / norm

    def normalize(self):
        if self.P_wolf.sum()>0:
            self.P_wolf /= self.P_wolf.sum()
        if self.P_seer.sum()>0:
            self.P_seer /= self.P_seer.sum()

    def update_on_seer_claim(self, claimer_id: int):
        """当有人声明自己是预言家时更新信念"""
        #self.claimed_seers.add(claimer_id)
        self.claimed_seers[claimer_id] = 0.5

        # 如果我是真预言家，且有人冒充，极大提高对其狼人嫌疑
        if self.role == Role.SEER and claimer_id != self.self_id:
            self.P_wolf[claimer_id] = min(1.0, self.P_wolf[claimer_id] + 0.8)

    def update_on_accuse_me(self, speaker: int, is_claimed_seer: bool):
        """当有人指控自己时，根据对方身份更新信念"""
        if self.role != Role.WOLF:  # 我不是狼人，被指控是冤枉的
            if is_claimed_seer:
                # 如果是自称预言家指控我，他很可能是狼人
                self.P_wolf[speaker] = min(1.0, self.P_wolf[speaker] + self.fake_seer_penalty)
            else:
                # 如果是普通角色指控我，提升较小的狼人嫌疑
                self.P_wolf[speaker] = min(1.0, self.P_wolf[speaker] + self.normal_accuse_penalty)

    def update_on_fake_accuse(self, speaker: int, target: int):
        """
        处理狼人伪装预言家的指控：
        - is_seer: 当前调用者是否是预言家（真预言家直接标记狼人）
        """
        # 被指控者（若为村民）直接标记指控者为狼
        if self.self_id == target and self.role != Role.WOLF:
            self.P_wolf[speaker] = 1.0

        # 其他玩家按普通指控更新目标嫌疑
        else:
            self.update_on_wolf_accuse(speaker, target)

    def update_on_support(self, supporter_id: int, target_id: int):
        """处理支持行为带来的信任关系变化"""
        # 通用支持行为处理（按角色区分）
        if target_id == self.self_id:  # 有人支持我
            if self.role == Role.SEER:
                # 我是预言家时：任何支持我的人都小幅可信
                self.P_wolf[supporter_id] = max(0.0,
                                                self.P_wolf[supporter_id] - self.trust_gain)

            elif self.role == Role.VILLAGER:
                # 我是村民时：
                if supporter_id in self.claimed_seers:
                    # 支持者是声称的预言家：显著增加信任
                    self.claimed_seers[supporter_id] = min(1.0,
                                                           self.claimed_seers[supporter_id] + self.trust_gain)
                    self.P_wolf[supporter_id] = max(0.0,
                                                    self.P_wolf[supporter_id] - self.trust_gain)
                else:
                    # 普通玩家支持我：小幅可信
                    self.P_wolf[supporter_id] = max(0.0,
                                                    self.P_wolf[supporter_id] - self.trust_gain * 0.5)


    def update_on_seer_check(self, target: int, is_wolf: bool):
        """
        处理预言家验人结果
        Args:
            target: 被验的玩家id
            is_wolf: 是否是狼人
        """
        if is_wolf:
            self.P_wolf[target] = 1.0  # 如果验出是狼人，直接标记为1
        else:
            self.P_wolf[target] = 0.0  # 如果验出是好人，直接标记为0
        #self.normalize()  # 更新后需要归一化

    def on_seer_death(self, dead_seer_id: int):
        """处理预言家死亡的连锁反应"""
        # 提高其他自称预言家的狼人嫌疑
        for claimer_id in self.claimed_seers:
            if claimer_id != dead_seer_id:
                self.P_wolf[claimer_id] = min(1.0,
                                              self.P_wolf[claimer_id] + self.seer_death_suspicion)

                # 同时提高该预言家支持过的人的嫌疑
                for target_id, supporters in self.supported_by.items():
                    if claimer_id in supporters:
                        self.P_wolf[target_id] = min(1.0,
                                                     self.P_wolf[target_id] + self.wolf_teammate_suspicion)
