import logging
import sys
from logging.handlers import RotatingFileHandler
# ---------- Logging 基础配置 ----------
LOG_LEVEL = logging.DEBUG          # DEBUG / INFO / WARNING / ERROR
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
DATE_FMT   = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=DATE_FMT,
    handlers=[logging.StreamHandler(sys.stdout)]  # 先给一个控制台
)

# ---------- 追加文件轮转 ----------
file_handler = RotatingFileHandler(
    "server_debug.log",  # 文件名
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=3,            # 最多保留 3 个旧文件
    encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FMT))
logging.getLogger().addHandler(file_handler)

# --------- 示例：减少某些 noisy 包的日志 ---------
logging.getLogger("engineio").setLevel(logging.WARNING)
logging.getLogger("socketio").setLevel(logging.INFO)
# import logging
# log = logging.getLogger(__name__)   # 名字 = 当前模块路径
log = logging.getLogger(__name__)



# Flask相关导入
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# 游戏相关导入
from werewolf_env.werewolf_env import WerewolfEnv, Role, TalkType
from agents.belief_agent import BeliefAgent
from utils.role_assigner import RoleAssigner
import json
import numpy as np

# 使用gevent作为异步后端
from gevent import monkey
monkey.patch_all()

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'werewolf-game-secret'
# 初始化SocketIO，使用threading模式，允许跨域访问
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

class GameState:
    """游戏状态类，管理整个游戏的状态和数据"""
    def __init__(self):
        self.reset()

    def reset(self):
        """重置游戏状态"""
        self.env = None
        self.players = {}  # {id: {"name": str, "role": Role}}
        self.num_players = 6  # 默认6人局
        self.num_wolves = 2   # 默认2狼
        self.num_seers = 1
        self.day = 0
        self.current_step = None
        self.current_speaker = 0  # 当前发言者
        self.night_actions = {}  # 存储夜晚行动
        self.vote_results = {}  # 存储投票结果
        self.death_record = []  # 存储死亡记录
        
        # 新增：更细粒度的步骤控制
        self.sub_step = None  # 子步骤：'wolf_action', 'seer_action', 'process_night'
        self.wolf_actions_collected = False
        self.seer_actions_collected = False
        
        # 遗言相关状态
        self.legacy_spoken = False  # 是否已经处理过遗言
        self.legacy_speakers = []   # 需要留遗言的玩家列表
        self.current_legacy_speaker = 0  # 当前遗言发言者索引
        self.first_night_legacy_done = False  # 是否已经处理过第一夜的遗言

    def initialize_game(self):
        """初始化游戏环境和角色"""
        try:
            # 计算村民数量（不包括预言家）
            num_villagers = self.num_players - self.num_wolves - self.num_seers  # 减1是因为有一个预言家
            # 使用 RoleAssigner 的静态方法分配角色
            roles = RoleAssigner.assign_roles(self.num_wolves, num_villagers)
            
            # 初始化游戏环境
            self.env = WerewolfEnv(roles)
            
            # 重置环境 - 这是关键步骤！
            obs, _ = self.env.reset()
            
            # 初始化玩家信息
            for i in range(self.num_players):
                self.players[i] = {
                    "name": f"玩家{i+1}",
                    "role": roles[i]
                }
                
            # 为每个玩家创建 AI 代理，使用与 demo_belief 相同的方式
            for i in range(self.num_players):
                agent = BeliefAgent(i, self.num_players, roles[i])
                self.env.add_agent(agent)
            
            # 设置初始状态
            self.current_step = 'night'
            self.env.stage = "night"
            self.day = 1
            
            # 初始化 BeliefAgent 的静态变量
            BeliefAgent.current_speaker = 0
            
            return roles
            
        except Exception as e:
            print(f"Error initializing game: {str(e)}")
            raise

    def sync_state(self, new_step):
        """同步游戏状态和环境状态"""
        self.current_step = new_step
        self.env.stage = new_step.lower()
        if new_step == 'night':
            self.night_actions.clear()
            self.vote_results.clear()

    def enter_night_phase(self):
        """进入夜晚阶段"""
        self.sync_state('night')
        # 重置夜晚相关状态
        self.sub_step = None
        self.wolf_actions_collected = False
        self.seer_actions_collected = False
        
    def enter_talk_phase(self):
        """进入发言阶段"""
        self.sync_state('talk')
        self.current_speaker = 0
        BeliefAgent.current_speaker = 0
        
    def enter_vote_phase(self):
        """进入投票阶段"""
        self.sync_state('vote')

    def is_game_over(self):
        """检查游戏是否结束"""
        return self.env._check_win() is not None
        
    def get_winner(self):
        """获取获胜方"""
        winner = self.env._check_win()
        if winner is None:
            return None
        return "狼人" if winner == "WOLF" else "好人"

def broadcast_host_message(message, phase=None):
    """广播主持人消息"""
    emit('host_message', {'message': message, 'phase': phase}, broadcast=True)

def broadcast_beliefs():
    """广播所有玩家的信念值"""
    try:
        beliefs = {}
        for player_id, player_info in game_state.players.items():
            agent = game_state.env.agents[player_id]
            if hasattr(agent, 'belief'):
                # 检查是否有 P_wolf 和 P_seer 属性
                p_wolf = agent.belief.P_wolf.tolist() if hasattr(agent.belief, 'P_wolf') else None
                p_seer = agent.belief.P_seer.tolist() if hasattr(agent.belief, 'P_seer') else None
                
                if p_wolf is not None or p_seer is not None:
                    beliefs[player_id] = {
                        'p_wolf': p_wolf if p_wolf is not None else [0] * game_state.num_players,
                        'p_seer': p_seer if p_seer is not None else [0] * game_state.num_players
                    }
        emit('update_beliefs', {'beliefs': beliefs}, broadcast=True)
    except Exception as e:
        print(f"Error broadcasting beliefs: {str(e)}")
        broadcast_host_message("信念广播出错")

def broadcast_wait_next(next_step):
    """广播等待下一步的消息"""
    emit('wait_next', {'next_step': next_step}, broadcast=True)

def update_all_beliefs(events):
    """更新所有玩家的信念并广播"""
    try:
        for agent in game_state.env.agents:
            if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                # 更新信念
                agent.strategies['belief_update'].execute(game_state.env, events)
                
                # 归一化概率
                if hasattr(agent, 'belief'):
                    if hasattr(agent.belief, 'normalize'):
                        agent.belief.normalize()
        broadcast_beliefs()
    except Exception as e:
        print(f"Error updating beliefs: {str(e)}")
        broadcast_host_message("信念更新出错")

def broadcast_death(victim_id):
    """广播死亡信息"""
    if victim_id >= 0 and victim_id < game_state.num_players:
        victim = game_state.players[victim_id]
        game_state.death_record.append({
            "day": game_state.day,
            "victim": victim_id,
            "role": victim["role"].name
        })
        broadcast_host_message(f"{victim['name']}（{victim['role'].name}）死亡", "death_announce")

def broadcast_vote_results():
    """广播投票结果"""
    vote_summary = {}
    for voter_id, target_id in game_state.vote_results.items():
        if target_id < game_state.env.N:  # 只统计有效投票
            if target_id not in vote_summary:
                vote_summary[target_id] = []
            vote_summary[target_id].append(voter_id)
    
    # 按票数排序
    sorted_results = sorted(vote_summary.items(), key=lambda x: len(x[1]), reverse=True)
    
    # 广播投票结果
    for target_id, voters in sorted_results:
        voter_names = [game_state.players[int(v)]["name"] for v in voters]
        target_name = game_state.players[target_id]["name"]
        broadcast_host_message(
            f"{target_name} 获得 {len(voters)} 票（来自：{', '.join(voter_names)}）",
            "vote_result"
        )

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@socketio.on('start_game')
def on_start_game(data):
    print("[FLASK] 收到 start_game:", data)
    """
    data = {
        "num_players": 6,
        "num_wolves": 2,
        "assign_mode": "random" | "manual",
        "roles":  [0,1,2,1,0,1],      # 仅当 manual
        "strategy": "belief" | "random"
    }
    """
    try:
        # 1) 读取参数并简单校验
        num_players = int(data.get("num_players", 6))
        num_wolves  = int(data.get("num_wolves", 2))
        num_Seers   = int(data.get("num_Seers", 1))
        assign_mode = data.get("assign_mode", "random")
        roles_list  = data.get("roles", [])
        strategy    = data.get("strategy", "belief")

        if not (4 <= num_players <= 12):
            emit("error", {"msg": "玩家人数必须 4-12"})
            return

        game_state.reset()
        game_state.num_players = num_players
        game_state.num_wolves  = num_wolves
        game_state.num_seers  = num_Seers

        # 2) 生成角色
        if assign_mode == "manual":
            if len(roles_list) != num_players:
                emit("error", {"msg": "手动角色数量与玩家数量不符"})
                return
            roles = [Role(r) for r in roles_list]
        else:                               # random
            num_villagers = num_players - num_wolves - 1
            roles = RoleAssigner.assign_roles(num_wolves, num_villagers)

        # 3) 初始化环境与智能体
        game_state.env = WerewolfEnv(roles)
        obs, _ = game_state.env.reset()

        for i in range(num_players):
            game_state.players[i] = {"name": f"玩家{i+1}", "role": roles[i]}
            # if strategy == "random":
            #     agent = RandomAgent(i, num_players, roles[i])
            # else:
            if strategy == "belief":
                agent = BeliefAgent(i, num_players, roles[i])
            game_state.env.add_agent(agent)

        # 4) 其余流程同之前……
        game_state.current_step = 'night'
        game_state.day = 1
        BeliefAgent.current_speaker = 0

        emit('game_started', {
            'players': [
                {"id": pid, "name": p["name"], "role": p["role"].name, "beliefs": {}}
                for pid, p in game_state.players.items()
            ]
        }, broadcast=True)

        broadcast_beliefs()
        game_state.enter_night_phase()
        broadcast_host_message(f"第 {game_state.day} 天开始", "day_start")
        broadcast_host_message("天黑请闭眼", "night")
        broadcast_wait_next("进入夜晚")

    except Exception:
        logging.exception("Error starting game with custom settings")
        broadcast_host_message("游戏启动出错，请检查设置")


# ---------------- 新增 ----------------
@socketio.on('end_game')
def on_end_game():
    """强制结束当前游戏"""
    try:
        if not game_state.env:    # 尚未开始
            return
        # 标记已结束，方便前端直接展示结果
        winner = game_state.get_winner() or "未完成"
        emit('game_over', {
            "winner": winner,
            "roles": {pid: pinfo["role"].name for pid, pinfo in game_state.players.items()}
        }, broadcast=True)
        # 简单重置
        game_state.reset()
        print("[INFO] 游戏被手动终止")
    except Exception:
        logging.exception("Error while ending game")
# --------------------------------------


@socketio.on('next_step')
def on_next_step():
    """处理游戏进程"""
    try:
        if game_state.current_step == 'night':
            # 夜晚阶段 - 分步骤控制
            game_state.env.stage = "night"
            
            if game_state.sub_step is None:
                # 开始狼人行动
                game_state.sub_step = 'wolf_action'
                broadcast_host_message("天黑请闭眼，狼人请睁眼", "wolf_turn")
                broadcast_wait_next("狼人选择目标")
                
            elif game_state.sub_step == 'wolf_action' and not game_state.wolf_actions_collected:
                # 收集狼人行动
                try:
                    log.debug("开始收集狼人行动")
                    for i in range(game_state.num_players):
                        if game_state.env.alive[i]:
                            player_info = game_state.players[i]
                            if player_info["role"] == Role.WOLF:
                                agent = game_state.env.agents[i]
                                action = agent.act(game_state.env)
                                action_val = int(action[0]) if isinstance(action, np.ndarray) else int(action)
                                game_state.night_actions[str(i)] = action
                                
                                log.debug("狼人%s(%s) -> %s", i, player_info["name"],
                                            action_val if action_val < game_state.env.N else "无目标")

                                
                                if action_val < game_state.env.N:
                                    target_name = game_state.players[action_val]["name"]
                                    broadcast_host_message(f"狼人 {player_info['name']} 选择杀害 {target_name}", "wolf_turn")
                                else:
                                    broadcast_host_message(f"狼人 {player_info['name']} 选择不杀人", "wolf_turn")
                    
                    log.debug("--- 狼人行动已收集 ---")
                    for wid, act in game_state.night_actions.items():
                        if game_state.players[int(wid)]["role"] == Role.WOLF:
                            tgt = int(act[0]) if isinstance(act, np.ndarray) else int(act)
                            log.debug("狼人%-2s 刀向 %s", wid, tgt if tgt < game_state.env.N else "无目标")

                    game_state.wolf_actions_collected = True
                    broadcast_host_message("狼人请闭眼", "wolf_turn")
                    broadcast_wait_next("预言家行动")
                    
                except Exception as e:
                    print(f"Error collecting wolf actions: {str(e)}")
                    broadcast_host_message("狼人行动出错")
                    
            elif game_state.sub_step == 'wolf_action' and game_state.wolf_actions_collected:
                # 开始预言家行动
                game_state.sub_step = 'seer_action'
                broadcast_host_message("预言家请睁眼", "seer_turn")
                broadcast_wait_next("预言家选择目标")
                
            elif game_state.sub_step == 'seer_action' and not game_state.seer_actions_collected:
                # 收集预言家行动
                try:
                    for i in range(game_state.num_players):
                        if game_state.env.alive[i]:
                            player_info = game_state.players[i]
                            if player_info["role"] == Role.SEER:
                                agent = game_state.env.agents[i]
                                action = agent.act(game_state.env)
                                game_state.night_actions[str(i)] = action
                                
                                action_val = int(action[0]) if isinstance(action, np.ndarray) else int(action)
                                if action_val < game_state.env.N:
                                    target_name = game_state.players[action_val]["name"]
                                    broadcast_host_message(f"预言家 {player_info['name']} 选择查验 {target_name}", "seer_turn")
                                else:
                                    broadcast_host_message(f"预言家 {player_info['name']} 选择不查验", "seer_turn")
                            elif player_info["role"] == Role.VILLAGER:
                                # 村民无夜晚行动
                                game_state.night_actions[str(i)] = game_state.env.N
                    # ---------- 预言家行动收集完成后 DEBUG ----------
                    log.debug("--- 预言家行动已收集 ---")
                    for sid, act in game_state.night_actions.items():
                        # 只打印预言家
                        if game_state.players[int(sid)]["role"] == Role.SEER:
                            tgt = int(act[0]) if isinstance(act, np.ndarray) else int(act)
                            log.debug("预言家%-2s 查验 %s",
                                    sid,
                                    tgt if tgt < game_state.env.N else "无目标")
                    # -----------------------------------------------

                    game_state.seer_actions_collected = True
                    broadcast_host_message("预言家请闭眼", "seer_turn")
                    broadcast_wait_next("处理夜晚结果")
                    
                except Exception as e:
                    print(f"Error collecting seer actions: {str(e)}")
                    broadcast_host_message("预言家行动出错")
                    
            elif game_state.sub_step == 'seer_action' and game_state.seer_actions_collected:
                # 处理夜晚结果
                game_state.sub_step = 'process_night'
                
                try:
                    # 执行环境步进
                    log.debug("执行环境.step, 夜晚行动: %s", game_state.night_actions)
                    obs, rewards, terms, truncs, info = game_state.env.step(game_state.night_actions)
                    log.debug("环境.step 完成, 存活: %s", np.where(game_state.env.alive)[0].tolist())
                    
                    # 处理预言家查验结果
                    seer_events = []
                    night_events = []
                    
                    if hasattr(game_state.env, 'event_log') and game_state.env.event_log:
                        log.debug("检查最近5个事件")
                        for event in game_state.env.event_log[-5:]:
                            log.debug("事件内容: %s", event)
                            if isinstance(event, dict):
                                if event.get("phase") == "seer_check":
                                    seer_id = event.get("seer")
                                    target_id = event.get("target")
                                    role_checked = event.get("role")
                                    if seer_id is not None and target_id is not None:
                                        seer_name = game_state.players[seer_id]["name"]
                                        target_name = game_state.players[target_id]["name"]
                                        result = "狼人" if role_checked == Role.WOLF else "好人"
                                        broadcast_host_message(f"预言家 {seer_name} 查验了 {target_name}，结果是{result}", "seer_turn")
                                        
                                        # 立即为预言家更新信念
                                        seer_agent = game_state.env.agents[seer_id]
                                        if hasattr(seer_agent, 'belief') and hasattr(seer_agent.belief, 'update_on_seer_check'):
                                            is_wolf = (role_checked == Role.WOLF)
                                            seer_agent.belief.update_on_seer_check(target_id, is_wolf)
                                        
                                        seer_events.append(('seer_check', seer_id, target_id, role_checked))
                                
                                elif event.get("phase") == "night" and "victim" in event:
                                    victim_id = event.get("victim")
                                    print(f"[DEBUG] 发现夜晚死亡事件: victim_id={victim_id}")
                                    if victim_id is not None and victim_id >= 0:
                                        victim_name = game_state.players[victim_id]["name"]
                                        broadcast_host_message(f"最终结果：{victim_name} 被狼人杀害", "night_end")
                                        broadcast_death(victim_id)
                                        night_events.append(('night', None, None, victim_id, -1))
                                    elif victim_id == -1:
                                        broadcast_host_message("狼人没有达成一致，今晚没有人死亡", "night_end")
                    
                    # 处理预言家死亡逻辑
                    if night_events:
                        for event in night_events:
                            victim_id = event[3]
                            if victim_id >= 0:
                                # 检查死者是否声称过预言家
                                for agent in game_state.env.agents:
                                    if (hasattr(agent, 'belief') and 
                                        hasattr(agent.belief, 'claimed_seers') and 
                                        victim_id in agent.belief.claimed_seers):
                                        # 调用预言家死亡处理
                                        if hasattr(agent, 'strategies') and agent.strategies.get('belief_update'):
                                            agent.strategies['belief_update']._handle_seer_death(game_state.env)
                    
                    # 广播信念更新
                    broadcast_beliefs()
                    
                    # 检查是否需要处理第一夜死亡玩家的遗言
                    if game_state.day == 1 and not getattr(game_state, 'first_night_legacy_done', False):
                        first_night_victims = []
                        # 查找第一夜（day=0）死亡的玩家
                        for event in game_state.env.event_log:
                            if (isinstance(event, dict) and 
                                event.get("phase") == "night" and 
                                event.get("day") == 0 and  # 只检查第一夜
                                event.get("victim") is not None and 
                                event.get("victim") >= 0):
                                victim_id = event.get("victim")
                                if victim_id < game_state.num_players and not game_state.env.alive[victim_id]:
                                    if victim_id not in first_night_victims:
                                        first_night_victims.append(victim_id)
                        
                        print(f"[DEBUG] 第一夜死者需要遗言：{first_night_victims}")
                        
                        # 处理第一夜死者的遗言
                        if first_night_victims:
                            game_state.first_night_legacy_done = True
                            broadcast_host_message("第一夜的死者留下遗言：", "legacy")
                            
                            for victim_id in first_night_victims:
                                victim_player = game_state.players[victim_id]
                                
                                # 让死亡的agent发言
                                agent = game_state.env.agents[victim_id]
                                action = agent.act(game_state.env)
                                
                                # 收集该玩家的所有遗言事件并整合显示
                                legacy_events = []
                                if hasattr(game_state.env, 'event_log') and game_state.env.event_log:
                                    for event in game_state.env.event_log:
                                        if (isinstance(event, dict) and 
                                            event.get("phase") == "legacy" and 
                                            event.get("speaker") == victim_id):
                                            legacy_events.append(event)
                                
                                # 整合遗言内容为一条消息
                                if legacy_events:
                                    legacy_parts = []
                                    speaker_name = victim_player["name"]
                                    is_seer_claim = False
                                    
                                    for event in legacy_events:
                                        talk_type = event.get("type")
                                        target_id = event.get("target")
                                        role_info = event.get("role")
                                        
                                        if talk_type == TalkType.CLAIM_SEER:
                                            legacy_parts.insert(0, "我是预言家")  # 身份声明放在最前面
                                            is_seer_claim = True
                                        elif talk_type == TalkType.ACCUSE:
                                            if target_id is not None and target_id >= 0:
                                                target_name = game_state.players[target_id]["name"]
                                                if role_info == Role.WOLF:
                                                    check_day = event.get("check_day", 0)
                                                    legacy_parts.append(f"第{check_day}天晚上查验{target_name}是狼人")
                                                else:
                                                    legacy_parts.append(f"{target_name}是狼人")
                                        elif talk_type == TalkType.SUPPORT:
                                            if target_id is not None and target_id >= 0:
                                                target_name = game_state.players[target_id]["name"]
                                                if role_info == Role.VILLAGER:
                                                    check_day = event.get("check_day", 0)
                                                    legacy_parts.append(f"第{check_day}天晚上查验{target_name}是好人")
                                                else:
                                                    legacy_parts.append(f"{target_name}是好人")
                                        elif talk_type == TalkType.CLAIM_GOOD:
                                            if not is_seer_claim:  # 如果没有声称预言家才说我是好人
                                                legacy_parts.append("我是好人")
                                    
                                    # 组合遗言消息
                                    if legacy_parts:
                                        legacy_message = f"【遗言】{speaker_name}：{'，'.join(legacy_parts)}。"
                                        message_type = "talk_seer_claim" if is_seer_claim else "talk"
                                        broadcast_host_message(legacy_message, message_type)
                                    else:
                                        broadcast_host_message(f"【遗言】{speaker_name}保持沉默。", "talk")
                            
                            # 再次广播信念更新
                            broadcast_beliefs()
                    
                    # 检查游戏是否结束
                    if game_state.is_game_over():
                        winner = game_state.get_winner()
                        broadcast_host_message(f"游戏结束！{winner}获胜！", "end")
                        return
                    
                    # 重置夜晚状态，进入发言阶段
                    game_state.sub_step = None
                    game_state.wolf_actions_collected = False
                    game_state.seer_actions_collected = False
                    game_state.night_actions.clear()
                    
                    game_state.current_step = 'talk'
                    game_state.current_speaker = 0
                    BeliefAgent.current_speaker = 0
                    broadcast_host_message("天亮了，开始发言阶段", "talk")
                    broadcast_wait_next("第一位玩家发言")
                    
                except Exception as e:
                    print(f"Error processing night results: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    broadcast_host_message("夜晚结果处理出错")
                
        elif game_state.current_step == 'talk':
            # 发言阶段 - 逐个玩家发言控制
            
            log.debug("发言阶段开始: day=%s, current_speaker=%s",
                game_state.day, game_state.current_speaker)

            
            # 正常发言阶段
            current_player = game_state.players[game_state.current_speaker]
            
            if game_state.env.alive[game_state.current_speaker]:
                # 让当前玩家发言
                broadcast_host_message(f"请 {current_player['name']} 发言", "talk")
                
                # 让 agent 自己处理发言逻辑
                agent = game_state.env.agents[game_state.current_speaker]
                action = agent.act(game_state.env)
                
                # 从环境的 event_log 中获取最新的发言事件并显示
                if hasattr(game_state.env, 'event_log') and game_state.env.event_log:
                    latest_events = game_state.env.event_log[-3:]  # 获取最近的几个事件
                    for event in latest_events:
                        if (isinstance(event, dict) and 
                            event.get("phase") == "talk" and 
                            event.get("speaker") == game_state.current_speaker):
                            
                            speaker_name = current_player["name"]
                            talk_type = event.get("type")
                            target_id = event.get("target")
                            role_info = event.get("role")
                            
                            if talk_type == TalkType.CLAIM_SEER:
                                broadcast_host_message(f"{speaker_name} 声称自己是预言家！", "talk_seer_claim")
                            elif talk_type == TalkType.ACCUSE:
                                if target_id is not None and target_id >= 0:
                                    target_name = game_state.players[target_id]["name"]
                                    if role_info == Role.WOLF:
                                        check_day = event.get("check_day", game_state.day - 1)
                                        broadcast_host_message(f"{speaker_name} 说：第{check_day}天晚上我查验了{target_name}，结果是狼人！", "talk")
                                    else:
                                        broadcast_host_message(f"{speaker_name} 指控 {target_name} 是狼人！", "talk")
                            elif talk_type == TalkType.SUPPORT:
                                if target_id is not None and target_id >= 0:
                                    target_name = game_state.players[target_id]["name"]
                                    if role_info == Role.VILLAGER:
                                        check_day = event.get("check_day", game_state.day - 1)
                                        broadcast_host_message(f"{speaker_name} 说：第{check_day}天晚上我查验了{target_name}，结果是好人。", "talk")
                                    else:
                                        broadcast_host_message(f"{speaker_name} 认为 {target_name} 是好人。", "talk")
                            elif talk_type == TalkType.CLAIM_GOOD:
                                broadcast_host_message(f"{speaker_name} 说：我是好人。", "talk")
                    
                # 广播信念更新
                broadcast_beliefs()
            else:
                broadcast_host_message(f"{current_player['name']} 已死亡，跳过发言", "talk")
            
            # 移动到下一个发言者
            game_state.current_speaker += 1
            BeliefAgent.current_speaker += 1
            
            # 跳过已死亡的玩家，找到下一个活着的玩家
            while (game_state.current_speaker < game_state.num_players and 
                   not game_state.env.alive[game_state.current_speaker]):
                game_state.current_speaker += 1
                BeliefAgent.current_speaker += 1
            
            if game_state.current_speaker < game_state.num_players:
                # 还有活着的玩家需要发言
                next_player = game_state.players[game_state.current_speaker]
                broadcast_wait_next(f"下一位：{next_player['name']} 发言")
            else:
                # 所有活着的玩家都发言完毕，进入投票阶段
                game_state.enter_vote_phase()
                broadcast_host_message("发言阶段结束，开始投票", "vote")
                broadcast_wait_next("开始投票")
                
        elif game_state.current_step == 'vote':
    # ---------- 投票阶段 ----------
            try:
                actions = {}
                game_state.vote_results = {}          # ← 清空上一轮记录

                # 1) 收集动作 & 生成 vote_results
                for i in range(game_state.num_players):
                    if game_state.env.alive[i]:
                        agent   = game_state.env.agents[i]
                        action  = agent.act(game_state.env)                   # ndarray 或 int
                        act_int = int(action[0]) if isinstance(action, np.ndarray) else int(action)
                        actions[str(i)] = action
                        game_state.vote_results[i] = act_int

                # 2) DEBUG 打印
                log.debug("--- 投票动作 ---")
                for pid, tgt in game_state.vote_results.items():
                    log.debug("玩家%-2s → %s", pid, tgt if tgt < game_state.env.N else "弃票")

                votes = np.zeros(game_state.env.N, dtype=int)
                for tgt in game_state.vote_results.values():
                    if 0 <= tgt < game_state.env.N:
                        votes[tgt] += 1
                log.debug("票数统计: %s", votes.tolist())

                # 3) 广播玩家投票信息
                for pid, tgt in game_state.vote_results.items():
                    voter_name = game_state.players[pid]["name"]
                    if 0 <= tgt < game_state.env.N:
                        target_name = game_state.players[tgt]["name"]
                        broadcast_host_message(f"{voter_name} 投票给了 {target_name}", "vote")
                    else:
                        broadcast_host_message(f"{voter_name} 弃票", "vote")

                broadcast_vote_results()
                
                # 执行环境步进
                obs, rewards, terms, truncs, info = game_state.env.step(actions)
                
                # 从环境的 event_log 中获取投票结果
                if hasattr(game_state.env, 'event_log') and game_state.env.event_log:
                    for event in game_state.env.event_log[-2:]:
                        if isinstance(event, dict) and event.get("phase") == "vote":
                            victim_id = event.get("out")
                            if victim_id is not None and victim_id >= 0:
                                broadcast_death(victim_id)
                            elif victim_id == -1:
                                broadcast_host_message("平票，没有人被投死", "vote_result")
                
                # 广播信念更新
                broadcast_beliefs()
                
                # 检查游戏是否结束
                if game_state.is_game_over():
                    winner = game_state.get_winner()
                    broadcast_host_message(f"游戏结束！{winner}获胜！", "end")
                    return
                
                # 进入下一天夜晚
                game_state.enter_night_phase()
                game_state.day += 1
                broadcast_host_message(f"第 {game_state.day} 天开始", "day_start")
                broadcast_wait_next("进入夜晚")
                
            except Exception as e:
                print(f"Error in vote phase: {str(e)}")
                import traceback
                traceback.print_exc()
                broadcast_host_message("游戏进程出错，请刷新页面重试")
                
    except Exception as e:
        print(f"Error in game process: {str(e)}")
        import traceback
        traceback.print_exc()
        broadcast_host_message("游戏进程出错，请刷新页面重试")

# 全局游戏状态
game_state = GameState()

if __name__ == '__main__':
    socketio.run(app, debug=True) 