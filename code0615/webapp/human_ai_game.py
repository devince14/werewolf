import logging
from flask_socketio import emit
from werewolf_env.werewolf_env import Role, TalkType
import numpy as np

log = logging.getLogger(__name__)

class HumanAIGameHandler:
    """人机对战模式处理器"""
    
    def __init__(self, game_state):
        self.game_state = game_state
        self.human_player_id = None  # 人类玩家ID
        self.waiting_for_human = False  # 是否在等待人类玩家操作
        self.current_action_type = None  # 当前等待的操作类型
        
    def set_human_player(self, player_id):
        """设置人类玩家ID"""
        self.human_player_id = player_id
        
    def is_human_turn(self, player_id):
        """检查是否是人类玩家的回合"""
        return player_id == self.human_player_id
        
    def broadcast_human_action_request(self, action_type, options=None):
        """向人类玩家请求行动"""
        self.waiting_for_human = True
        self.current_action_type = action_type
        
        human_player = self.game_state.players[self.human_player_id]
        
        emit('human_action_request', {
            'action_type': action_type,
            'player_name': human_player['name'],
            'options': options or []
        }, broadcast=True)
        
    def handle_night_phase_human(self):
        """处理夜晚阶段的人类玩家行动"""
        if not self.game_state.env.alive[self.human_player_id]:
            return False  # 人类玩家已死亡，跳过
            
        human_player = self.game_state.players[self.human_player_id]
        
        if human_player["role"] == Role.WOLF:
            # 人类是狼人，需要选择击杀目标
            alive_targets = []
            for i in range(self.game_state.num_players):
                if (self.game_state.env.alive[i] and 
                    i != self.human_player_id and 
                    self.game_state.players[i]["role"] != Role.WOLF):
                    alive_targets.append({
                        'id': i,
                        'name': self.game_state.players[i]['name']
                    })
            
            alive_targets.append({'id': self.game_state.env.N, 'name': '不杀人'})
            
            self.broadcast_human_action_request('wolf_kill', alive_targets)
            return True
            
        elif human_player["role"] == Role.SEER:
            # 人类是预言家，需要选择查验目标
            alive_targets = []
            for i in range(self.game_state.num_players):
                if (self.game_state.env.alive[i] and 
                    i != self.human_player_id):
                    alive_targets.append({
                        'id': i,
                        'name': self.game_state.players[i]['name']
                    })
            
            alive_targets.append({'id': self.game_state.env.N, 'name': '不查验'})
            
            self.broadcast_human_action_request('seer_check', alive_targets)
            return True
            
        return False  # 村民在夜晚无行动
        
    def handle_talk_phase_human(self):
        """处理发言阶段的人类玩家行动"""
        if not self.game_state.env.alive[self.human_player_id]:
            return False
            
        # 为人类玩家提供发言选项
        talk_options = [
            {'type': 'claim_seer', 'text': '声称预言家'},
            {'type': 'claim_good', 'text': '声称好人'},
            {'type': 'accuse', 'text': '指控某人是狼'},
            {'type': 'support', 'text': '支持某人是好人'},
            {'type': 'pass', 'text': '保持沉默'}
        ]
        
        # 获取所有玩家作为目标选项（包括死亡的，因为预言家可能查验过死者）
        all_targets = []
        for i in range(self.game_state.num_players):
            if i != self.human_player_id:
                status = "存活" if self.game_state.env.alive[i] else "已死"
                all_targets.append({
                    'id': i,
                    'name': f"{self.game_state.players[i]['name']}({status})"
                })
        
        # 只获取活着的玩家作为指控/支持目标
        alive_targets = []
        for i in range(self.game_state.num_players):
            if (self.game_state.env.alive[i] and i != self.human_player_id):
                alive_targets.append({
                    'id': i,
                    'name': self.game_state.players[i]['name']
                })
        
        self.broadcast_human_action_request('talk', {
            'talk_options': talk_options,
            'all_targets': all_targets,  # 所有玩家（用于预言家查验结果）
            'alive_targets': alive_targets,  # 活着的玩家（用于指控/支持）
            'check_results': ['wolf', 'good']  # 预言家查验结果选项
        })
        return True


    def handle_last_words_human(self):
        """处理人类玩家的遗言阶段"""
        if self.game_state.env.alive[self.human_player_id]:
            return False

        talk_options = [
            {'type': 'claim_seer', 'text': '声称预言家'},
            {'type': 'claim_good', 'text': '声称好人'},
            {'type': 'accuse', 'text': '指控某人是狼'},
            {'type': 'support', 'text': '支持某人是好人'},
            {'type': 'pass', 'text': '保持沉默'}
        ]

        all_targets = []
        for i in range(self.game_state.num_players):
            if i != self.human_player_id:
                status = "存活" if self.game_state.env.alive[i] else "已死"
                all_targets.append({'id': i,
                                   'name': f"{self.game_state.players[i]['name']}({status})"})

        alive_targets = []
        for i in range(self.game_state.num_players):
            if self.game_state.env.alive[i] and i != self.human_player_id:
                alive_targets.append({'id': i,
                                     'name': self.game_state.players[i]['name']})

        self.broadcast_human_action_request('last_words', {
            'talk_options': talk_options,
            'all_targets': all_targets,
            'alive_targets': alive_targets,
            'check_results': ['wolf', 'good']
        })
        return True  
    def handle_vote_phase_human(self):
        """处理投票阶段的人类玩家行动"""
        if not self.game_state.env.alive[self.human_player_id]:
            return False
            
        # 为人类玩家提供投票选项
        alive_targets = []
        for i in range(self.game_state.num_players):
            if (self.game_state.env.alive[i] and i != self.human_player_id):
                alive_targets.append({
                    'id': i,
                    'name': self.game_state.players[i]['name']
                })
        
        alive_targets.append({'id': self.game_state.env.N, 'name': '弃票'})
        
        self.broadcast_human_action_request('vote', alive_targets)
        return True
        
    def process_human_action(self, action_data):
        """处理人类玩家的行动"""
        if not self.waiting_for_human:
            return False
            
        action_type = action_data.get('action_type')
        target_id = action_data.get('target_id')
        talk_type = action_data.get('talk_type')
        check_result = action_data.get('check_result')  # 预言家查验结果
        
        try:
            if action_type == 'wolf_kill':
                # 处理狼人击杀
                self.game_state.night_actions[str(self.human_player_id)] = target_id
                emit('host_message', {
                    'message': f"你选择了{'击杀 ' + self.game_state.players[target_id]['name'] if target_id < self.game_state.env.N else '不杀人'}",
                    'phase': 'wolf_turn'
                }, broadcast=True)
                
            elif action_type == 'seer_check':
                # 处理预言家查验
                self.game_state.night_actions[str(self.human_player_id)] = target_id
                emit('host_message', {
                    'message': f"你选择了{'查验 ' + self.game_state.players[target_id]['name'] if target_id < self.game_state.env.N else '不查验'}",
                    'phase': 'seer_turn'
                }, broadcast=True)
                
            elif action_type == 'talk':
                # 处理发言
                self._process_human_talk(talk_type, target_id, check_result)
                
            elif action_type == 'last_words':
                # 处理遗言
                self._process_human_legacy(talk_type, target_id, check_result)

            elif action_type == 'vote':
                # 处理投票
                self.game_state.vote_results[self.human_player_id] = target_id
                emit('host_message', {
                    'message': f"你投票给了{'弃票' if target_id >= self.game_state.env.N else self.game_state.players[target_id]['name']}",
                    'phase': 'vote'
                }, broadcast=True)
                
            self.waiting_for_human = False
            self.current_action_type = None
            
            # 通知前端继续游戏流程
            emit('human_action_completed', {}, broadcast=True)
            
            # 广播最新的信念状态
            self._broadcast_beliefs()
            
            # 根据不同阶段继续游戏流程
            if action_type == 'talk':
                self._continue_talk_phase()
            elif action_type in ['wolf_kill', 'seer_check']:
                self._continue_night_phase()
            elif action_type == 'last_words':
                # 遗言不影响游戏流程，直接继续
                pass
            elif action_type == 'vote':
                self._continue_vote_phase()
            
            return True
            
        except Exception as e:
            log.error(f"处理人类行动时出错: {e}")
            emit('error', {'message': '操作失败，请重试'})
            return False
            
    def _process_human_talk(self, talk_type, target_id, check_result=None):
        """处理人类玩家的发言"""
        human_player = self.game_state.players[self.human_player_id]
        speaker_name = human_player['name']
        
        # 构造发言事件并添加到环境日志
        event = {
            "phase": "talk",
            "speaker": self.human_player_id,
            "type": None,
            "target": target_id,
            "role": None
        }
        
        message = ""
        
        if talk_type == 'claim_seer':
            event["type"] = TalkType.CLAIM_SEER
            
            if target_id is not None and target_id >= 0 and check_result is not None:
                # 预言家声称并公布查验结果
                target_name = self.game_state.players[target_id]['name']
                result_text = "狼人" if check_result == "wolf" else "好人"
                role_int = Role.WOLF if check_result == "wolf" else Role.VILLAGER
                event["role"] = role_int
                event["check_day"] = self.game_state.day - 1  # 通常是前一天晚上查验的
                message = f"{speaker_name} 声称自己是预言家！昨晚我查验了{target_name}，结果是{result_text}！"
                emit('host_message', {'message': message, 'phase': 'talk_seer_claim'}, broadcast=True)
                
                # 先添加预言家声称到 public_log
                claim_entry = (0, self.human_player_id, int(TalkType.CLAIM_SEER), -1, -1)
                self.game_state.env.public_log.append(claim_entry)
                
                # 如果是真预言家，更新环境的验人记录
                if human_player["role"] == Role.SEER:
                    self.game_state.env.seer_records.append((
                        self.game_state.day - 1,  # 查验日期
                        target_id,                # 查验目标
                        role_int                  # 查验结果
                    ))
                    print(f"[DEBUG] Added seer check record: day={self.game_state.day-1}, target={target_id}, result={check_result}")
                
                # 再添加查验结果到 public_log
                check_talk_type = TalkType.ACCUSE if check_result == "wolf" else TalkType.SUPPORT
                check_entry = (0, self.human_player_id, int(check_talk_type), target_id, role_int)
                self.game_state.env.public_log.append(check_entry)
                
                # 处理两个事件的信念更新
                self._update_beliefs_from_events([claim_entry, check_entry])
                
            else:
                # 只是声称预言家身份
                message = f"{speaker_name} 声称自己是预言家！"
                emit('host_message', {'message': message, 'phase': 'talk_seer_claim'}, broadcast=True)
                
                # 添加预言家声称到 public_log
                log_entry = (0, self.human_player_id, int(TalkType.CLAIM_SEER), -1, -1)
                self.game_state.env.public_log.append(log_entry)
                
                # 更新信念
                self._update_beliefs_from_events([log_entry])
            
        elif talk_type == 'claim_good':
            event["type"] = TalkType.CLAIM_GOOD
            message = f"{speaker_name} 说：我是好人。"
            emit('host_message', {'message': message, 'phase': 'talk'}, broadcast=True)
            # 添加到 public_log
            log_entry = (0, self.human_player_id, int(TalkType.CLAIM_GOOD), -1, -1)
            self.game_state.env.public_log.append(log_entry)
            self._update_beliefs_from_events([log_entry])
            
        elif talk_type == 'accuse' and target_id is not None:
            event["type"] = TalkType.ACCUSE
            event["role"] = Role.WOLF
            target_name = self.game_state.players[target_id]['name']
            message = f"{speaker_name} 指控 {target_name} 是狼人！"
            emit('host_message', {'message': message, 'phase': 'talk'}, broadcast=True)
            # 添加到 public_log
            log_entry = (0, self.human_player_id, int(TalkType.ACCUSE), target_id, Role.WOLF)
            self.game_state.env.public_log.append(log_entry)
            self._update_beliefs_from_events([log_entry])
            
        elif talk_type == 'support' and target_id is not None:
            event["type"] = TalkType.SUPPORT
            event["role"] = Role.VILLAGER
            target_name = self.game_state.players[target_id]['name']
            message = f"{speaker_name} 认为 {target_name} 是好人。"
            emit('host_message', {'message': message, 'phase': 'talk'}, broadcast=True)
            # 添加到 public_log
            log_entry = (0, self.human_player_id, int(TalkType.SUPPORT), target_id, Role.VILLAGER)
            self.game_state.env.public_log.append(log_entry)
            self._update_beliefs_from_events([log_entry])
            
        elif talk_type == 'pass':
            message = f"{speaker_name} 保持沉默。"
            emit('host_message', {'message': message, 'phase': 'talk'}, broadcast=True)
            return  # 沉默不添加事件
        
        # 添加事件到环境日志（仅用于记录）

        if hasattr(self.game_state.env, 'event_log'):
            self.game_state.env.event_log.append(event)

    def _process_human_legacy(self, talk_type, target_id, check_result=None):
        """处理人类玩家的遗言"""
        human_player = self.game_state.players[self.human_player_id]
        speaker_name = human_player['name']

        event = {
            "phase": "legacy",
            "speaker": self.human_player_id,
            "type": None,
            "target": target_id,
            "role": None
        }

        prefix = "【遗言】"

        message = ""

        if talk_type == 'claim_seer':
            event["type"] = TalkType.CLAIM_SEER
            if target_id is not None and target_id >= 0 and check_result is not None:
                target_name = self.game_state.players[target_id]['name']
                result_text = "狼人" if check_result == "wolf" else "好人"
                role_int = Role.WOLF if check_result == "wolf" else Role.VILLAGER
                event["role"] = role_int
                event["check_day"] = self.game_state.day - 1
                message = f"{prefix}{speaker_name} 声称自己是预言家！昨晚我查验了{target_name}，结果是{result_text}！"
                emit('host_message', {'message': message, 'phase': 'talk_seer_claim'}, broadcast=True)

                claim_entry = (0, self.human_player_id, int(TalkType.CLAIM_SEER), -1, -1)
                self.game_state.env.public_log.append(claim_entry)

                if human_player["role"] == Role.SEER:
                    self.game_state.env.seer_records.append((
                        self.game_state.day - 1,
                        target_id,
                        role_int
                    ))

                check_talk_type = TalkType.ACCUSE if check_result == "wolf" else TalkType.SUPPORT
                check_entry = (0, self.human_player_id, int(check_talk_type), target_id, role_int)
                self.game_state.env.public_log.append(check_entry)

                self._update_beliefs_from_events([claim_entry, check_entry])
            else:
                message = f"{prefix}{speaker_name} 声称自己是预言家！"
                emit('host_message', {'message': message, 'phase': 'talk_seer_claim'}, broadcast=True)
                log_entry = (0, self.human_player_id, int(TalkType.CLAIM_SEER), -1, -1)
                self.game_state.env.public_log.append(log_entry)
                self._update_beliefs_from_events([log_entry])

        elif talk_type == 'claim_good':
            event["type"] = TalkType.CLAIM_GOOD
            message = f"{prefix}{speaker_name} 说：我是好人。"
            emit('host_message', {'message': message, 'phase': 'talk'}, broadcast=True)
            log_entry = (0, self.human_player_id, int(TalkType.CLAIM_GOOD), -1, -1)
            self.game_state.env.public_log.append(log_entry)
            self._update_beliefs_from_events([log_entry])

        elif talk_type == 'accuse' and target_id is not None:
            event["type"] = TalkType.ACCUSE
            event["role"] = Role.WOLF
            target_name = self.game_state.players[target_id]['name']
            message = f"{prefix}{speaker_name} 指控 {target_name} 是狼人！"
            emit('host_message', {'message': message, 'phase': 'talk'}, broadcast=True)
            log_entry = (0, self.human_player_id, int(TalkType.ACCUSE), target_id, Role.WOLF)
            self.game_state.env.public_log.append(log_entry)
            self._update_beliefs_from_events([log_entry])

        elif talk_type == 'support' and target_id is not None:
            event["type"] = TalkType.SUPPORT
            event["role"] = Role.VILLAGER
            target_name = self.game_state.players[target_id]['name']
            message = f"{prefix}{speaker_name} 认为 {target_name} 是好人。"
            emit('host_message', {'message': message, 'phase': 'talk'}, broadcast=True)
            log_entry = (0, self.human_player_id, int(TalkType.SUPPORT), target_id, Role.VILLAGER)
            self.game_state.env.public_log.append(log_entry)
            self._update_beliefs_from_events([log_entry])

        elif talk_type == 'pass':
            message = f"{prefix}{speaker_name}保持沉默。"
            emit('host_message', {'message': message, 'phase': 'talk'}, broadcast=True)
            return

        if hasattr(self.game_state.env, 'event_log'):
            self.game_state.env.event_log.append(event)    
    def _update_beliefs_from_events(self, events):
        """根据事件列表更新AI玩家的信念"""
        try:
            # 为每个AI玩家更新信念
            from app import update_all_beliefs     # ← import 相对路径请按实际工程放置
            update_all_beliefs(events)
            # for agent in self.game_state.env.agents:
            #     if (hasattr(agent, 'strategies') and 
            #         agent.strategies.get('belief_update') and
            #         agent.agent_id != self.human_player_id):  # 不更新人类玩家的信念
                    
            #         # 调用信念更新策略，传入所有事件
            #         agent.strategies['belief_update'].execute(self.game_state.env, events)
            
            # # 广播更新后的信念
            # self._broadcast_beliefs()
            
        except Exception as e:
            log.error(f"更新信念时出错: {e}")
            
    def _broadcast_beliefs(self):
        """广播信念值 - 直接调用主应用函数"""
        # 直接调用全局的broadcast_beliefs函数
        # import sys
        # if 'webapp.app' in sys.modules:
        #     # 获取主应用模块中的函数
        #     main_module = sys.modules['webapp.app']
        import sys
        main_module = sys.modules.get('app')   # 统一按真实模块名查找
        if hasattr(main_module, 'broadcast_beliefs'):
            main_module.broadcast_beliefs()
            return
        
        # 备用方案：使用本地实现
        from flask_socketio import emit
        try:
            beliefs = {}
            for player_id, player_info in self.game_state.players.items():
                agent = self.game_state.env.agents[player_id]
                if hasattr(agent, 'belief'):
                    # 检查是否有 P_wolf 和 P_seer 属性
                    p_wolf = agent.belief.P_wolf.tolist() if hasattr(agent.belief, 'P_wolf') else None
                    p_seer = agent.belief.P_seer.tolist() if hasattr(agent.belief, 'P_seer') else None
                    
                    if p_wolf is not None or p_seer is not None:
                        beliefs[player_id] = {
                            'p_wolf': p_wolf if p_wolf is not None else [0] * self.game_state.num_players,
                            'p_seer': p_seer if p_seer is not None else [0] * self.game_state.num_players
                        }
            
            emit('update_beliefs', {'beliefs': beliefs}, broadcast=True)
        except Exception as e:
            log.error(f"人机模式广播信念时出错: {e}")
        
    def _continue_talk_phase(self):
        """继续发言阶段流程"""
        try:
            from flask_socketio import emit
            
            # 移动到下一个发言者
            self.game_state.current_speaker += 1
            
            # 更新BeliefAgent的current_speaker
            from agents.belief_agent import BeliefAgent
            BeliefAgent.current_speaker += 1
            
            # 跳过已死亡的玩家，找到下一个活着的玩家
            while (self.game_state.current_speaker < self.game_state.num_players and 
                   not self.game_state.env.alive[self.game_state.current_speaker]):
                self.game_state.current_speaker += 1
                BeliefAgent.current_speaker += 1
            
            if self.game_state.current_speaker < self.game_state.num_players:
                # 还有活着的玩家需要发言，广播等待下一个玩家
                next_player = self.game_state.players[self.game_state.current_speaker]
                emit('wait_next', {'next_step': f"下一位：{next_player['name']} 发言"}, broadcast=True)
            else:
                # 所有活着的玩家都发言完毕，进入投票阶段
                self.game_state.enter_vote_phase()
                emit('host_message', {'message': "发言阶段结束，开始投票", 'phase': 'vote'}, broadcast=True)
                emit('wait_next', {'next_step': "开始投票"}, broadcast=True)
                
        except Exception as e:
            log.error(f"继续发言阶段时出错: {e}")
    
    def _continue_night_phase(self):
        """继续夜晚阶段流程"""
        try:
            from flask_socketio import emit
            
            # 根据当前子阶段继续流程
            if self.game_state.sub_step == 'wolf_action':
                # 人类狼人行动完成，现在需要收集AI狼人的行动并显示汇总
                self._collect_and_display_wolf_actions()
                self.game_state.wolf_actions_collected = True
                emit('host_message', {'message': "狼人请闭眼", 'phase': 'wolf_turn'}, broadcast=True)
                emit('wait_next', {'next_step': "预言家行动"}, broadcast=True)
            elif self.game_state.sub_step == 'seer_action':
                self.game_state.seer_actions_collected = True
                emit('host_message', {'message': "预言家请闭眼", 'phase': 'seer_turn'}, broadcast=True)
                emit('wait_next', {'next_step': "处理夜晚结果"}, broadcast=True)
                
        except Exception as e:
            log.error(f"继续夜晚阶段时出错: {e}")
            
    def _collect_and_display_wolf_actions(self):
        """收集并显示所有狼人的行动选择"""
        try:
            from flask_socketio import emit
            import numpy as np
            
            # 收集AI狼人的行动
            for i in range(self.game_state.num_players):
                if (self.game_state.env.alive[i] and 
                    self.game_state.players[i]["role"] == Role.WOLF and
                    i != self.human_player_id):  # 不是人类玩家
                    
                    agent = self.game_state.env.agents[i]
                    action = agent.act(self.game_state.env)
                    action_val = int(action[0]) if isinstance(action, np.ndarray) else int(action)
                    self.game_state.night_actions[str(i)] = action
                    
                    log.debug(f"AI狼人{i}({self.game_state.players[i]['name']}) -> {action_val if action_val < self.game_state.env.N else '无目标'}")
            
            # 收集所有狼人的目标选择并显示汇总
            wolf_targets = []
            wolf_choices = {}
            
            for wid, act in self.game_state.night_actions.items():
                if self.game_state.players[int(wid)]["role"] == Role.WOLF:
                    tgt = int(act[0]) if isinstance(act, np.ndarray) else int(act)
                    wolf_choices[int(wid)] = tgt
                    if tgt < self.game_state.env.N:
                        wolf_targets.append(tgt)
            
            # 显示狼人团队选择汇总
            emit('host_message', {'message': "=== 狼人团队选择汇总 ===", 'phase': 'wolf_turn'}, broadcast=True)
            for wolf_id, target_id in wolf_choices.items():
                wolf_name = self.game_state.players[wolf_id]["name"]
                if target_id < self.game_state.env.N:
                    target_name = self.game_state.players[target_id]["name"]
                    emit('host_message', {'message': f"狼人 {wolf_name} 选择击杀 {target_name}", 'phase': 'wolf_turn'}, broadcast=True)
                else:
                    emit('host_message', {'message': f"狼人 {wolf_name} 选择不杀人", 'phase': 'wolf_turn'}, broadcast=True)
            
            # 显示最终决策逻辑
            if len(wolf_targets) == 0:
                emit('host_message', {'message': "所有狼人都选择不杀人，今晚无人死亡", 'phase': 'wolf_turn'}, broadcast=True)
            elif len(set(wolf_targets)) == 1:
                # 所有狼人选择同一目标
                final_target = wolf_targets[0]
                target_name = self.game_state.players[final_target]["name"]
                emit('host_message', {'message': f"狼人团队一致选择击杀 {target_name}", 'phase': 'wolf_turn'}, broadcast=True)
            else:
                # 狼人选择不同目标，需要随机决定
                from collections import Counter
                target_counts = Counter(wolf_targets)
                max_votes = max(target_counts.values())
                most_voted = [t for t, c in target_counts.items() if c == max_votes]
                
                if len(most_voted) == 1:
                    final_target = most_voted[0]
                    target_name = self.game_state.players[final_target]["name"]
                    emit('host_message', {'message': f"根据投票结果，最终选择击杀 {target_name}", 'phase': 'wolf_turn'}, broadcast=True)
                else:
                    # 平票情况，随机选择
                    import random
                    final_target = random.choice(most_voted)
                    target_name = self.game_state.players[final_target]["name"]
                    candidates = [self.game_state.players[t]["name"] for t in most_voted]
                    emit('host_message', {'message': f"狼人选择出现平票（{', '.join(candidates)}），随机决定击杀 {target_name}", 'phase': 'wolf_turn'}, broadcast=True)
                    
        except Exception as e:
            log.error(f"收集和显示狼人行动时出错: {e}")
    
    def _continue_vote_phase(self):
        """继续投票阶段流程"""
        try:
            from flask_socketio import emit
            
            # 标记人类玩家已完成投票，避免重复投票
            self.game_state.human_vote_completed = True
            
            # 确保人类玩家的投票被记录到环境的event_log中
            if self.human_player_id in self.game_state.vote_results:
                human_vote = self.game_state.vote_results[self.human_player_id]
                # 将人类玩家的投票记录到事件日志中
                vote_event = {
                    "day": self.game_state.env.day,
                    "phase": "vote",
                    "voter": self.human_player_id,
                    "target": human_vote,
                    "alive": [i for i, a in enumerate(self.game_state.env.alive) if a]
                }
                self.game_state.env.event_log.append(vote_event)
                print(f"[DEBUG] Recorded human vote event: {vote_event}")
            
            # 投票阶段完成后，触发下一步处理
            emit('wait_next', {'next_step': "处理投票结果"}, broadcast=True)
                
        except Exception as e:
            log.error(f"继续投票阶段时出错: {e}") 

    # def _update_beliefs_from_human_talk(self, event):
    #     """根据人类玩家的发言更新AI玩家的信念（保留用于兼容）"""
    #     # 获取最后添加到 public_log 的事件
    #     if not self.game_state.env.public_log:
    #         return
        
    #     last_event = self.game_state.env.public_log[-1]
    #     self._update_beliefs_from_events([last_event]) 
    def _update_beliefs_from_events(self, events):
        """根据事件列表更新AI玩家的信念"""
        try:
            # 为每个AI玩家更新信念
            for agent in self.game_state.env.agents:
                if (hasattr(agent, 'strategies') and 
                    agent.strategies.get('belief_update') and
                    agent.agent_id != self.human_player_id):  # 不更新人类玩家的信念
                    
                    # 调用信念更新策略，传入所有事件
                    agent.strategies['belief_update'].execute(self.game_state.env, events)
            
            # 广播更新后的信念
            self._broadcast_beliefs()
            
        except Exception as e:
            log.error(f"更新信念时出错: {e}")
            
 