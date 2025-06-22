# -*- coding: utf-8 -*-
"""
PettingZoo 并行环境 — 狼人杀简化版
依赖：
    pip install gymnasium pettingzoo numpy
"""

from enum import IntEnum
from typing import Dict, List, Tuple
import random
import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

# 阶段编码
PHASE_TALK = 0
PHASE_LEGACY = 1
PHASE_NIGHT = 2

class Role(IntEnum):
    WOLF = 0
    VILLAGER = 1
    SEER = 2

class TalkType(IntEnum):
    CLAIM_GOOD = 0
    ACCUSE     = 1
    SUPPORT    = 2
    CLAIM_SEER = 3  # 新增：声明自己是预言家

ROLE_LIST = [Role.WOLF, Role.VILLAGER, Role.SEER]

class WerewolfEnv(ParallelEnv):
    metadata = {"name": "werewolf_simplified"}

    def __init__(
        self,
        roles: List[Role],
        talk_history_len: int = 20,
        max_nights: int      = 10,
        seed: int | None     = 0,
    ):
        super().__init__()
        self.random = random.Random(seed)
        # 必须且仅有一个预言家
        if roles.count(Role.SEER) != 1:
            raise ValueError("roles 列表中必须恰好包含一个 Role.SEER")
        self.roles = roles
        self.N     = len(roles)
        self.history_len = talk_history_len
        self.max_nights  = max_nights
        self.agents = []  # 存储所有智能体实例

        # 行动空间
        self.talk_space = spaces.MultiDiscrete([2, 3, self.N+1])  # (TalkType, target)，target==N 表示无目标
        self.vote_space = spaces.Discrete(self.N+1)           # 0..N-1 投票对象，N 弃票
        self.kill_space = spaces.Discrete(self.N+1)           # 狼人夜晚 0..N-1 杀人，N 弃刀
        self.seer_space = spaces.Discrete(self.N+1)           # 预言家夜晚 0..N-1 查验，N 弃验
        self.noop_space = spaces.Discrete(1)                  # 其他角色无动作

        # 观测空间：增加 phase 字段
        # 观测空间：增加 phase 和 role_int 字段，history 扩成 5 列
        self.observation_space_template = spaces.Dict({
            "role":       spaces.MultiBinary(len(ROLE_LIST)),
            "alive_mask": spaces.MultiBinary(self.N),
            "talk_history": spaces.Box(
                low=-1, high=max(self.N, len(ROLE_LIST)),
                shape=(talk_history_len, 5),  # (phase, speaker, type, target, role_int)
                dtype=np.int32
            ),
        })
        # self.observation_space_template = spaces.Dict({
        #     "role":       spaces.MultiBinary(len(ROLE_LIST)),
        #     "alive_mask": spaces.MultiBinary(self.N),
        #     "talk_history": spaces.Box(
        #         low=-1, high=self.N,
        #         shape=(talk_history_len, 4),
        #         dtype=np.int32
        #     ),
        # })

        # self.agent_names = [str(i) for i in range(self.N)]

        # 状态变量
        self.alive       = None  # np.ndarray[bool]
        self.night_count = 0
        self.day         = 1
        self.stage       = "night"  # 修改初始阶段为 night
        # public_log 记录 (phase, speaker, type, target)
        self.public_log  = []
        # event_log 记录详细事件
        self.event_log   = []
        self.seer_records = []  # 存储所有历史查验结果

    def add_agent(self, agent):
        """添加智能体到环境中"""
        self.agents.append(agent)

    def _one_hot(self, role: Role):
        vec = np.zeros(len(ROLE_LIST), dtype=np.int32)
        vec[role] = 1
        return vec

    def _gen_obs(self, pid: int):
        role_vec   = self._one_hot(self.roles[pid])
        alive_mask = self.alive.astype(np.int32)
        history    = np.full((self.history_len, 5), -1, dtype=np.int32)
        tail       = self.public_log[-self.history_len:]
        if tail:
            history[-len(tail):] = np.array(tail, dtype=np.int32)
        return {"role": role_vec, "alive_mask": alive_mask, "talk_history": history}, {}

    def observation_space(self, agent):
        return self.observation_space_template

    def action_space(self, agent):
        idx = int(agent)
        role = self.roles[idx]
        if self.stage == "talk":
            return self.talk_space
        if self.stage == "vote":
            return self.vote_space
        if self.stage == "night":
            if role == Role.WOLF:
                return self.kill_space
            if role == Role.SEER:
                return self.seer_space
            # 村民无夜晚动作
            return self.noop_space
        return self.noop_space

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.random.seed(seed)
        self.alive       = np.ones(self.N, dtype=bool)
        self.night_count = 0
        self.day         = 0
        self.stage       = "night"
        self.public_log.clear()
        self.event_log.clear()
        self.seer_records = []  # 重置查验记录
        obs, info = {}, {}
        for i in range(self.N):
            obs[str(i)], info[str(i)] = self._gen_obs(i)
        return obs, info

    def _check_win(self):
        wolves_alive = np.any((np.array(self.roles)==Role.WOLF)&self.alive)
        good_alive   = np.any((np.array(self.roles)!=Role.WOLF)&self.alive)
        if not wolves_alive:
            return "GOOD"
        if not good_alive:
            return "WOLF"
        if self.night_count >= self.max_nights:
            return "WOLF"
        return None

    def step(self, action_dict: Dict[str, np.ndarray]):
        assert self.stage in {"talk","vote","night"}, "阶段错误"
        def alive_list(): return [i for i,a in enumerate(self.alive) if a]

        seer_idx = self.roles.index(Role.SEER)

        # TALK 阶段：直接进入投票阶段，因为日志已经在act中记录
        if self.stage == "talk":
            self.stage = "vote"

        elif self.stage == "vote":
            votes = np.zeros(self.N, dtype=int)
            for pid_str, v_ in action_dict.items():
                pid = int(pid_str)
                if not self.alive[pid]: continue
                # 确保获取单个整数值
                v = int(v_[0]) if isinstance(v_, np.ndarray) else int(v_)
                if 0 <= v < self.N and self.alive[v]:
                    votes[v] += 1
            maxv = votes.max()
            cnt_max = (votes==maxv).sum()
            out = -1
            if maxv>0 and cnt_max==1:
                out = int(votes.argmax())
                self.alive[out] = False
            self.event_log.append({
                "day":self.day,"phase":"vote",
                "out":out,"votes":votes.tolist(),
                "alive":alive_list()
            })
            self.stage = "night"

        elif self.stage == "night":
            day_night = self.day
            wolf_targets = []
            seer_target  = None
            seer_pid     = None          # ← ① 先定义
            
            try:
                # 预处理动作字典，确保所有动作都是有效的整数
                processed_actions = {}
                for pid_str, action in action_dict.items():
                    pid = int(pid_str)
                    try:
                        # 确保 pid 是字符串并且可以转换为整数
                        if not isinstance(pid_str, str):
                            continue
                        pid = int(pid_str)
                        
                        # 检查玩家是否存活
                        if not self.alive[pid]:
                            continue
                            
                        # 获取玩家角色
                        role = self.roles[pid]
                        
                        # 处理动作值
                        if isinstance(action, np.ndarray):
                            if action.size == 0:
                                continue
                            a = int(action[0])
                        else:
                            a = int(action)
                            
                        # 存储处理后的动作
                        processed_actions[pid] = (role, a)
                        
                    except (ValueError, TypeError, IndexError):
                        continue
                
                # 处理狼人的行动
                for pid, (role, action) in processed_actions.items():
                    if role == Role.WOLF and 0 <= action < self.N:
                        wolf_targets.append(action)
                
                # 处理预言家的行动
                for pid, (role, action) in processed_actions.items():
                    if role == Role.SEER and 0 <= action < self.N:
                        seer_target = action
                        seer_pid    = pid
                        break  # 只取第一个有效的预言家行动
                
                # 记录验人结果
                if seer_target is not None:
                    role_checked = int(self.roles[seer_target])
                    self.seer_records.append((day_night, seer_target, role_checked))
                    self.event_log.append({
                        "day": day_night,
                        "phase": "seer_check",
                        "seer": seer_pid,
                        "target": seer_target,
                        "role": role_checked,
                        "alive": alive_list()
                    })
                
                # 处理狼人击杀
                target = -1
                if wolf_targets:
                    # 统计每个目标被投票的次数
                    vote_counts = np.bincount(wolf_targets, minlength=self.N)
                    max_votes = vote_counts.max()
                    # 获取得票最多的目标
                    max_vote_targets = np.where(vote_counts == max_votes)[0]
                    if len(max_vote_targets) > 0:
                        # 如果有平票，随机选择一个
                        target = int(self.random.choice(max_vote_targets))
                
                # 执行击杀
                if 0 <= target < self.N and self.alive[target]:
                    self.alive[target] = False
                
                # 记录夜晚事件
                self.event_log.append({
                    "day": day_night,
                    "phase": "night",
                    "victim": target,
                    "alive": alive_list()
                })
                
            except Exception as e:
                print(f"[DEBUG] Error in night phase: {str(e)}")
                self.event_log.append({
                    "day": day_night,
                    "phase": "night",
                    "error": str(e),
                    "alive": alive_list()
                })
            
            self.night_count += 1
            self.day += 1
            self.stage = "talk"

        winner=self._check_win()
        terminated = winner is not None
        rewards = {str(i):0.0 for i in range(self.N)}
        if terminated:
            for i in range(self.N):
                good = (self.roles[i]!=Role.WOLF)
                if (winner=="GOOD" and good) or (winner=="WOLF" and not good):
                    rewards[str(i)] = 1.0
            self.event_log.append({
                "day":self.day,"phase":"end",
                "winner":winner,"alive":alive_list()
            })
            self.stage="end"

        self.god_view_info = {"roles":[int(r) for r in self.roles],"alive":alive_list()}
        obs,info={},{}
        for i in range(self.N):
            obs[str(i)],info[str(i)] = self._gen_obs(i)
        terms = {str(i):terminated for i in range(self.N)}
        truns = {str(i):False for i in range(self.N)}
        return obs, rewards, terms, truns, info

    def render(self, n_events=None, god=False):
        """渲染最近的n_events个事件"""
        print(f"Day {self.day} | Stage: {self.stage}")
        print(f"Alive: {[i for i,a in enumerate(self.alive) if a]}")
        if god:
            print(f"God view roles: {[int(r) for r in self.roles]}")
        print(f"Last {n_events} events:")
        for ev in self.event_log[-n_events:]:
            day = ev["day"]
            phase = ev["phase"]
            if phase == "talk":
                speaker = ev["speaker"]
                tt = TalkType(ev["type"]).name
                target = ev["target"]
                role_str = f" is {Role(ev['role']).name}" if ev.get("role") is not None else ""
                print(f"  D{day} TALK  P{speaker} {tt} {target if target>=0 else '-'}{role_str}")
            elif phase == "legacy":
                speaker = ev["speaker"]
                tt = TalkType(ev["type"]).name
                target = ev["target"]
                role_str = f" is {Role(ev['role']).name}" if ev.get("role") is not None else ""
                print(f"  D{day} LEGACY  P{speaker} {tt} {target}{role_str}")
            elif phase == "vote":
                out = ev["out"]
                votes = ev["votes"]
                print(f"  D{day} VOTE  out={out} votes={votes}")
            elif phase == "night":
                victim = ev["victim"]
                print(f"  D{day} NIGHT victim={victim}")
            elif phase == "seer_check":
                seer = ev["seer"]
                target = ev["target"]
                role_str = f" is {Role(ev['role']).name}" if ev.get("role") is not None else ""
                print(f"  D{day} NIGHT_CHECK  seer P{seer} checked P{target}{role_str}")
        print("-")
