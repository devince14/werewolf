# # agents/strategies/belief_agent.py
# agents/strategies/belief_agent.py

# agents/strategies/belief_agent.py

import random
import numpy as np
from competition_template.werewolf_env import Role, TalkType, WerewolfEnv
from competition_template.agents.base_agent import BaseAgent
from competition_template.agents.belief.belief_state import BeliefState
from competition_template.agents.strategies.talk.seer_talk_strategy import SeerTalkStrategy

PHASE_TALK = 0
PHASE_LEGACY = 1
PHASE_NIGHT = 2

class BeliefAgent(BaseAgent):
    """
    基于信念的智能体，使用策略模式处理不同阶段的行为。
    使用两条分布 P_wolf（谁是狼）和 P_seer（谁是预言家）：
      - Seer 夜间/白天根据 P_wolf 采样查人/投票/发言
      - Villager 白天/投票根据 P_wolf 采样支持/投票
      - Wolf 夜间根据 P_seer 采样杀人；白天依旧 CLAIM_GOOD
    """
    # 类变量，用于跟踪发言顺序
    current_speaker = 0  # 当前应该发言的玩家编号
    day = 0  # 当前天数

    def __init__(self, agent_id: int, num_agents: int, role: Role):
        super().__init__(agent_id, num_agents, role)
        self.role = role
        self.belief = BeliefState(num_agents, agent_id, role)
        self.tau_wolf = 0.6
        self.tau_info = 0.5     # 保存阈值
        self.has_claimed_seer = False  # 记录是否已经声称过预言家
        self.checked_players = set()  # 记录已查验过的玩家
        self.shared_wolf_target = None  # 新增共享目标存储

    @classmethod
    def reset_talk_state(cls):
        """重置发言状态"""
        cls.current_speaker = 0
        cls.day += 1

    def act(self, env):
        """根据当前阶段执行相应的策略"""
        # 获取并处理新事件，立即更新信念
        new_events = self.fetch_new_events(env)
        
        # 如果是talk阶段的第一个发言者，且有新的夜晚事件，处理死亡信息
        if (env.stage == "talk" and
            BeliefAgent.current_speaker == 0):
            # 检查是否有夜晚死亡事件
            night_death_events = [
                event for event in env.event_log 
                if isinstance(event, dict) and 
                event.get("phase") == "night" and 
                event.get("victim", -1) >= 0 and
                event.get("day") == env.day - 1  # 确保是当前天数的事件
            ]
            if night_death_events:
                log_entry = ('night', None, None, night_death_events[0].get("victim"), -1)
                # 让所有agent更新信念（处理死亡信息）
                for agent in env.agents:
                    if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                        agent.strategies['belief_update'].execute(env, [log_entry])
    
        # 根据当前阶段选择策略
        if env.stage == "talk":
            strategy = self.strategies.get('talk')
            if not strategy:
                return np.array([0, TalkType.CLAIM_GOOD, env.N])
            
            # 如果已经死亡，发表遗言
            if not env.alive[self.agent_id]:
                claim_seer, talk_type, target = strategy.execute(env)
                self._record_legacy(env, talk_type, target)
                return np.array([claim_seer, talk_type, target])
            
            # 如果不是当前发言者，返回无效动作
            if BeliefAgent.current_speaker != self.agent_id:
                return np.array([0, TalkType.CLAIM_GOOD, env.N])
            
            # 正常发言
            claim_seer, talk_type, target = strategy.execute(env)
            self.record_talk(env, claim_seer, talk_type, target)
            return np.array([claim_seer, talk_type, target])

        elif env.stage == "vote":
            strategy = self.strategies.get('vote')
            if not strategy:
                return np.array([env.N])
            target = strategy.execute(env)
            # 确保返回值是一个包含单个整数的 numpy 数组
            return np.array([int(target) if isinstance(target, (int, np.integer)) else env.N])

        elif env.stage == "night":
            strategy = self.strategies.get('night')
            if not strategy:
                return np.array([env.N])
            target = strategy.execute(env)
            # 确保返回值是一个包含单个整数的 numpy 数组
            return np.array([int(target) if isinstance(target, (int, np.integer)) else env.N])

        return np.array([env.N])

    def _record_legacy(self, env, talk_type: int, target: int):
        """记录遗言事件"""
        print(f"[DEBUG] _record_legacy被调用: agent_id={self.agent_id}, talk_type={talk_type}, target={target}")
        
        # 如果是真预言家且有验人结果，优先报告查验结果并亮身份
        if self.role == Role.SEER and hasattr(env, 'seer_records') and env.seer_records:
            # 先声称预言家身份
            claim_entry = (PHASE_TALK, self.agent_id, int(TalkType.CLAIM_SEER), -1, -1)
            env.public_log.append(claim_entry)
            env.event_log.append({
                "day": env.day,
                "phase": "legacy",
                "speaker": self.agent_id,
                "type": int(TalkType.CLAIM_SEER),
                "target": -1,
                "alive": [i for i,a in enumerate(env.alive) if a]
            })
            
            # 报告所有查验结果
            for day, tgt, role_int in env.seer_records:
                if day < env.day:  # 只报告之前天数的结果
                    result_talk_type = TalkType.ACCUSE if role_int == Role.WOLF else TalkType.SUPPORT
                    log_entry = (PHASE_TALK, self.agent_id, int(result_talk_type), tgt, role_int)
                    env.public_log.append(log_entry)
                    env.event_log.append({
                        "day": env.day,
                        "phase": "legacy",
                        "speaker": self.agent_id,
                        "type": int(result_talk_type),
                        "target": tgt,
                        "role": role_int,
                        "check_day": day,
                        "alive": [i for i,a in enumerate(env.alive) if a]
                    })
        else:
            # 非预言家按照正常发言策略
            log_entry = (PHASE_TALK, self.agent_id, int(talk_type), target, -1)
            env.public_log.append(log_entry)
            env.event_log.append({
                "day": env.day,
                "phase": "legacy",
                "speaker": self.agent_id,
                "type": int(talk_type),
                "target": target,
                "alive": [i for i,a in enumerate(env.alive) if a]
            })
        
        print(f"[DEBUG] 遗言事件已添加到事件日志")
        
        # 立即更新所有玩家的信念
        for agent in env.agents:
            if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                # 获取最新添加的事件进行信念更新
                recent_entries = env.public_log[-len([e for e in env.event_log if e.get('phase') == 'legacy' and e.get('speaker') == self.agent_id]):]
                agent.strategies['belief_update'].execute(env, recent_entries)

    def record_talk(self, env, claim_seer: int, talk_type: int, target: int):
        """记录发言到环境日志"""
        if target == env.N:
            target = -1

        def alive_list(): 
            return [i for i,a in enumerate(env.alive) if a]
            
        # 检查是否是遗言阶段
        # 只有第一夜死亡的玩家才能留遗言
        is_first_night_legacy = (not env.alive[self.agent_id] and env.day == 1)
        print(f"[DEBUG] 遗言检测: agent_id={self.agent_id}, alive={env.alive[self.agent_id]}, day={env.day}, is_legacy={is_first_night_legacy}")
                                   
        if is_first_night_legacy:
            print(f"[DEBUG] 玩家 {self.agent_id} 进入遗言模式（第一夜死亡）")
            # 如果是遗言阶段，使用专门的遗言记录函数
            self._record_legacy(env, talk_type, target)
            return

        # 处理预言家身份声明
        if claim_seer == 1:
            # 记录声称预言家
            log_entry = (PHASE_TALK, self.agent_id, int(TalkType.CLAIM_SEER), -1, -1)
            env.public_log.append(log_entry)
            env.event_log.append({
                "day": env.day,
                "phase": "talk",
                "speaker": self.agent_id,
                "type": int(TalkType.CLAIM_SEER),
                "target": -1,
                "alive": alive_list()
            })
            
            # 立即更新所有玩家的信念（声称预言家）
            for agent in env.agents:
                if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                    agent.strategies['belief_update'].execute(env, [log_entry])
            #print(env.seer_records)
            # 如果是真预言家且有验人结果，立即公布
            if self.role == Role.SEER and env.seer_records:
                # 第二天首次跳身份时，公布所有验人结果
                if env.day == 2:
                    # 收集所有需要公布的验人结果
                    results_to_report = []
                    for day, tgt, role_int in env.seer_records:
                        if day < env.day:  # 只报告之前天数的结果
                            results_to_report.append((day, tgt, role_int))
                    
                    # 按优先级排序：村民结果优先，然后是狼人结果
                    wolf_results = [(d, t, r) for d, t, r in results_to_report if r == Role.WOLF]
                    villager_results = [(d, t, r) for d, t, r in results_to_report if r == Role.VILLAGER]
                    
                    # 优先公布村民结果
                    for day, tgt, role_int in villager_results:
                        talktype = TalkType.SUPPORT
                        log_entry = (PHASE_TALK, self.agent_id, int(talktype), tgt, role_int)
                        env.public_log.append(log_entry)
                        env.event_log.append({
                            "day": int(env.day),
                            "phase": "talk",
                            "speaker": self.agent_id,
                            "type": int(talktype),
                            "target": tgt,
                            "role": role_int,
                            "check_day": day,  # 添加查验的天数
                            "alive": alive_list()
                        })
                        
                        # 立即更新所有玩家的信念（验人结果）
                        for agent in env.agents:
                            if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                                agent.strategies['belief_update'].execute(env, [log_entry])
                    
                    # 然后公布狼人结果
                    for day, tgt, role_int in wolf_results:
                        talktype = TalkType.ACCUSE
                        log_entry = (PHASE_TALK, self.agent_id, int(talktype), tgt, role_int)
                        env.public_log.append(log_entry)
                        env.event_log.append({
                            "day": int(env.day),
                            "phase": "talk",
                            "speaker": self.agent_id,
                            "type": int(talktype),
                            "target": tgt,
                            "role": role_int,
                            "check_day": day,  # 添加查验的天数
                            "alive": alive_list()
                        })
                        
                        # 立即更新所有玩家的信念（验人结果）
                        for agent in env.agents:
                            if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                                agent.strategies['belief_update'].execute(env, [log_entry])
                
                # 其他天数，只公布最新的验人结果
                elif env.seer_records[-1][0] == env.day - 1:
                    _, tgt, role_int = env.seer_records[-1]
                    talktype = TalkType.ACCUSE if role_int == Role.WOLF else TalkType.SUPPORT
                    log_entry = (PHASE_TALK, self.agent_id, int(talktype), tgt, role_int)
                    env.public_log.append(log_entry)
                    env.event_log.append({
                        "day": int(env.day),
                        "phase": "talk",
                        "speaker": self.agent_id,
                        "type": int(talktype),
                        "target": tgt,
                        "role": role_int,
                        "check_day": env.day - 1,  # 添加查验的天数
                        "alive": alive_list()
                    })
                    
                    # 立即更新所有玩家的信念（验人结果）
                    for agent in env.agents:
                        if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                            agent.strategies['belief_update'].execute(env, [log_entry])
            
            # 如果是狼人假跳预言家，且有指控或支持目标，立即执行
            elif self.role == Role.WOLF and talk_type in (TalkType.ACCUSE, TalkType.SUPPORT) and target != -1:
                # 根据talk_type设置role_int：ACCUSE为0，SUPPORT为1
                role_int = 1 if talk_type == TalkType.SUPPORT else 0
                log_entry = (PHASE_TALK, self.agent_id, int(talk_type), target, role_int)
                env.public_log.append(log_entry)
                env.event_log.append({
                    "day": env.day,
                    "phase": "talk",
                    "speaker": self.agent_id,
                    "type": int(talk_type),
                    "target": target,
                    "role": role_int,
                    "alive": alive_list()
                })
                
                # 立即更新所有玩家的信念（狼人的指控/支持）
                for agent in env.agents:
                    if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                        agent.strategies['belief_update'].execute(env, [log_entry])
        
        # 处理普通发言
        elif talk_type == TalkType.CLAIM_GOOD:
            log_entry = (PHASE_TALK, self.agent_id, int(TalkType.CLAIM_GOOD), -1, -1)
            env.public_log.append(log_entry)
            env.event_log.append({
                "day": env.day,
                "phase": "talk",
                "speaker": self.agent_id,
                "type": int(TalkType.CLAIM_GOOD),
                "target": -1,
                "alive": alive_list()
            })
            
            # 立即更新所有玩家的信念
            for agent in env.agents:
                if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                    agent.strategies['belief_update'].execute(env, [log_entry])
                    
        elif talk_type in (TalkType.ACCUSE, TalkType.SUPPORT):
            # 如果是预言家的验人结果，记录role_int
            role_int = -1
            if (self.role == Role.SEER and env.seer_records 
                and env.seer_records[-1][0] == env.day - 1 
                and target == env.seer_records[-1][1]):
                role_int = env.seer_records[-1][2]
                
            log_entry = (PHASE_TALK, self.agent_id, int(talk_type), target, role_int)
            env.public_log.append(log_entry)
            env.event_log.append({
                "day": env.day,
                "phase": "talk",
                "speaker": self.agent_id,
                "type": int(talk_type),
                "target": target,
                "role": role_int if role_int != -1 else None,
                "alive": alive_list()
            })
            
            # 立即更新所有玩家的信念
            for agent in env.agents:
                if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                    agent.strategies['belief_update'].execute(env, [log_entry])

    def _get_wolf_teammates(self, env):
        """获取狼队友列表"""
        try:
            print(f"[DEBUG] Getting wolf teammates for agent {self.agent_id}")
            
            # 检查必要的属性
            if not hasattr(env, 'roles'):
                print("[DEBUG] env.roles does not exist")
                return []
            
            if not hasattr(env, 'N'):
                print("[DEBUG] env.N does not exist")
                return []
            
            if env.roles is None:
                print("[DEBUG] env.roles is None")
                return []
            
            # 获取狼队友列表
            teammates = []
            for i in range(env.N):
                try:
                    if (i != self.agent_id and 
                        env.roles[i] == Role.WOLF):
                        teammates.append(i)
                except (IndexError, TypeError) as e:
                    print(f"[DEBUG] Error checking player {i}: {str(e)}")
                    continue
                
            print(f"[DEBUG] Found wolf teammates: {teammates}")
            return teammates
        
        except Exception as e:
            print(f"[DEBUG] Error in _get_wolf_teammates: {str(e)}")
            return []

    def fetch_new_events(self, env: WerewolfEnv) -> list:
        """获取新事件并更新信念"""
        all_events = env.public_log
        new_events = all_events[self._last_event_idx:]
        self._last_event_idx = len(all_events)
        
        # 如果是预言家，检查验人结果
        if self.role == Role.SEER and env.seer_records:
            last = env.seer_records[-1]

            if len(last) == 4:               # 新格式 (day, tgt, role, seer_id)
                day, tgt, role_checked, who = last
            else:                            # 旧格式兼容  (day, tgt, role)
                day, tgt, role_checked      = last
                who = self.agent_id          # 先假设就是自己；或直接 return

            # 只有“我”验的才更新
            if who == self.agent_id and day == env.day:
                self.belief.update_on_seer_check(tgt, role_checked == Role.WOLF)
        
        # 更新信念
        # if new_events and self.strategies['belief_update']:
        #     self.strategies['belief_update'].execute(env, new_events)
            
        return new_events  # 返回新事件列表


