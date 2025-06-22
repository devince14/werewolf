# -*- coding: utf-8 -*-
"""
rl_team_selfplay.py  ——  PPO 自博弈训练脚本（随机角色顺序版）
─────────────────────────────────────────────────────────────
• 角色配比：1 狼 / 1 预言家 / 2 村民（每局随机打乱顺序）
• 狼队共享一个 PPO，好人（Seer+Villager×2）共享一个 PPO
• TALK / VOTE / NIGHT 均交由 PPO 决策
• 交替训练：先训狼队，对手随机；再训好人，对手用最新狼队
"""

import os, sys
from typing import List, Dict

import numpy as np
import gymnasium as gym
from pettingzoo.utils.wrappers import BaseWrapper  # 为类型提示
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import TensorBoardOutputFormat

# ---------- 项目内 import ----------
# 根据你的实际目录结构，保证能找到 werewolf_env 和 utils
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)

from werewolf_env.werewolf_env import WerewolfEnv, Role, TalkType
from utils.role_assigner import RoleAssigner

# ─────────────────────────────────────────────────────────────
#  一、TeamEnv：把并行 WerewolfEnv → 单智能体 Gym 环境
# ─────────────────────────────────────────────────────────────
class TeamEnv(gym.Env):
    """
    每次 reset():
        • 用 RoleAssigner 随机生成角色顺序
        • 新建 WerewolfEnv
        • 确定本阵营 pid 列表（狼队 / 好人队）
    动作空间：统一离散维度 ACTION_DIM
        [ TALK_SIZE | vote | kill | seer | noop ]
    观测：本阵营所有 agent 观测 flatten 后拼接
    """
    metadata = {}

    # ---------------------- 初始化 ---------------------- #
    def __init__(
        self,
        num_wolves: int,
        num_villagers: int,
        team: str,                          # "wolves" | "villagers"
        opponent_policy: PPO | None = None,
        seed: int | None = None,
        max_nights: int = 10,
        talk_history_len: int = 20,
    ):
        super().__init__()
        assert team in ("wolves", "villagers")
        self.is_wolf_team   = (team == "wolves")
        self.num_wolves     = num_wolves
        self.num_villagers  = num_villagers
        self.opponent_policy = opponent_policy
        self.max_nights      = max_nights
        self.talk_hist_len   = talk_history_len

        self.rng = np.random.default_rng(seed)

        # —— 用一次随机角色来 sample 空间 —— #
        init_roles = RoleAssigner.assign_roles(num_wolves, num_villagers)
        probe_env  = WerewolfEnv(
            roles=init_roles,
            talk_history_len=talk_history_len,
            max_nights=max_nights,
            seed=seed,
        )
        self.N = len(init_roles)

        # —— 统一动作维度 —— #
        self.TALK_SIZE  = 2 * 3 * (self.N + 1)          # phase(2) × type(3) × target(N+1)
        self.PHASE_SIZE = self.N + 1                    # vote / kill / seer
        self.ACTION_DIM = self.TALK_SIZE + 3 * self.PHASE_SIZE + 1  # +noop

        # —— 观测维度：用本阵营任一 pid 的 obs 来计算 —— #
        sample_obs_dict, _ = probe_env.reset()
        sample_pid = 0
        flat_sample = self._flatten_obs(sample_obs_dict[str(sample_pid)])
        # 先用 wolves 阵营估计最大拼接长度；真正 obs 在 reset() 时动态拼接
        team_len     = len(self._team_pids(init_roles))          # 1（狼）或 3（好人）
        obs_dim      = len(flat_sample) * team_len
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete(
            [self.ACTION_DIM] * team_len
        )

        # —— 首次 build 环境 —— #
        self._build_env(init_roles)

    # ---------------------- 工具函数 ---------------------- #
    @staticmethod
    def _flatten_obs(obs_dict: dict) -> np.ndarray:
        return np.concatenate(
            [
                obs_dict["role"].astype(np.float32).ravel(),
                obs_dict["alive_mask"].astype(np.float32).ravel(),
                obs_dict["talk_history"].astype(np.float32).ravel(),
            ]
        )

    def _team_pids(self, roles: List[Role]) -> List[int]:
        return [
            i
            for i, r in enumerate(roles)
            if (r == Role.WOLF) == self.is_wolf_team
        ]

    # ---------------------- 重新生成环境 ---------------------- #
    def _build_env(self, roles: List[Role]):
        self.raw = WerewolfEnv(
            roles=roles,
            talk_history_len=self.talk_hist_len,
            max_nights=self.max_nights,
            seed=self.rng.integers(1 << 30),
        )
        self.roles = roles
        self.team_pids = self._team_pids(roles)

    # ---------------------- 动作解码 ---------------------- #
    def _decode(self, phase: str, a: int, role: Role):
        if phase == "talk":
            a %= self.TALK_SIZE
            ph_flag, rem = divmod(a, 3 * (self.N + 1))
            tt, tgt      = divmod(rem, self.N + 1)
            return np.array([ph_flag, tt, tgt], dtype=np.int32)

        if phase == "vote":
            return a % self.PHASE_SIZE

        if phase == "night":
            if role in (Role.WOLF, Role.SEER):
                return a % self.PHASE_SIZE
            return 0  # 村民夜晚 noop

        return 0

    # ---------------------- Gym API ---------------------- #
    def reset(self, *, seed=None, options=None):
        new_roles = RoleAssigner.assign_roles(self.num_wolves, self.num_villagers)
        self._build_env(new_roles)

        obs_dict, _ = self.raw.reset()
        self._last_obs = obs_dict
        return self._make_obs(obs_dict), {}

    def step(self, action):
        phase = self.raw.stage
        full_act: Dict[str, int | np.ndarray] = {}

        # —— 1) 本阵营动作 —— #
        for i, pid in enumerate(self.team_pids):
            a = int(action[i]) % self.ACTION_DIM
            decoded = self._decode(phase, a, self.roles[pid])
            full_act[str(pid)] = decoded

        # —— 2) 对手动作 —— #
        opp_pids = [p for p in range(self.N) if p not in self.team_pids]
        if phase in ("talk", "vote", "night") and self.opponent_policy is not None:
            opp_flat = np.concatenate(
                [self._flatten_obs(self._last_obs[str(p)]) for p in opp_pids],
                dtype=np.float32,
            )
            opp_act_vec, _ = self.opponent_policy.predict(opp_flat, deterministic=True)
            for j, pid in enumerate(opp_pids):
                dec = self._decode(phase, int(opp_act_vec[j]), self.roles[pid])
                full_act[str(pid)] = dec
        else:
            for pid in opp_pids:
                full_act[str(pid)] = self.raw.action_space(str(pid)).sample()

        # —— 3) 环境一步 —— #
        obs_dict, rew_d, term_d, trunc_d, info = self.raw.step(full_act)
        self._last_obs = obs_dict
        flat_obs = self._make_obs(obs_dict)

        # —— 4) 奖励（胜负 + 轻度塑形） —— #
        base_reward = sum(
            rew_d[p] for p in rew_d if int(p) in self.team_pids
        )

        shaped = 0.0
        is_wolf_env = self.roles[self.team_pids[0]] == Role.WOLF
        seer_pid = next(i for i, r in enumerate(self.roles) if r == Role.SEER)

        # 4-1 Night
        if phase == "night" and len(self.raw.event_log) >= 1:
            ev_n = self.raw.event_log[-1]
            ev_s = self.raw.event_log[-2] if len(self.raw.event_log) >= 2 else {}
            victim = ev_n.get("victim", -1)
            # Seer check 奖励
            if not is_wolf_env and ev_s.get("phase") == "seer_check":
                tgt_role = ev_s["role"]
                shaped += 0.5 if tgt_role == Role.WOLF.value else -0.1
            # 狼人击杀奖励
            if is_wolf_env and victim >= 0:
                if victim == self.team_pids[0]:
                    shaped -= 1.0
                else:
                    if victim == seer_pid:
                        shaped += 0.5
                    elif self.roles[victim] != Role.WOLF:
                        shaped += 0.2
                    else:
                        shaped -= 0.2

        # 4-2 Vote
        if phase == "vote" and len(self.raw.event_log) >= 1:
            ev_v = self.raw.event_log[-1]
            if ev_v.get("phase") == "vote":
                out = ev_v["out"]
                if out >= 0:
                    if is_wolf_env:
                        shaped += 0.1 if self.roles[out] != Role.WOLF else -0.1
                    else:
                        shaped += 0.2 if self.roles[out] == Role.WOLF else -0.2

        # 4-3 终局
        terminal = any(term_d.values()) or any(trunc_d.values())
        if terminal and base_reward > 0:
            shaped += 1.0

        # 4-4 归一化
        max_step = (
            len(self.team_pids) + 0.5 + 0.2 + 1.0
        )  # 粗略上界
        total_reward = (base_reward + shaped) / max_step

        return flat_obs, total_reward, terminal, False, info

    # ---------------------- 观测拼接 ---------------------- #
    def _make_obs(self, obs_dict):
        vecs = [self._flatten_obs(obs_dict[str(pid)]) for pid in self.team_pids]
        return np.concatenate(vecs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
#  二、辅助：关闭 TensorBoard Writer（Win 文件锁）
# ─────────────────────────────────────────────────────────────
def close_tb_writer(model: PPO):
    for fmt in model.logger.output_formats:
        if isinstance(fmt, TensorBoardOutputFormat):
            try:
                fmt.writer.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
#  三、主流程
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ===== 配置 =====
    NUM_WOLVES, NUM_VILLAGERS = 1, 2        # 1 狼 + 2 村民（环境里自动再加 1 预言家）
    TIMESTEPS = 50_000                      # 每轮学习步数
    ROUNDS    = 10                          # 交替轮数

    # ===== 创建包装环境 =====
    wolf_env = Monitor(
        TeamEnv(NUM_WOLVES, NUM_VILLAGERS, team="wolves",    seed=0)
    )
    vill_env = Monitor(
        TeamEnv(NUM_WOLVES, NUM_VILLAGERS, team="villagers", seed=1)
    )

    # ===== 初始化 PPO =====
    wolf_model = PPO(
        "MlpPolicy",
        wolf_env,
        verbose=1,
        tensorboard_log="./logs/wolves",
        ent_coef=0.01,          # 鼓励探索 TALK 动作
    )
    vill_model = PPO(
        "MlpPolicy",
        vill_env,
        verbose=1,
        tensorboard_log="./logs/villagers",
        ent_coef=0.01,
    )

    # ===== 交替训练 =====
    for r in range(1, ROUNDS + 1):
        print(f"\n>>> Round {r}/{ROUNDS}: Train Wolves <<<")
        wolf_model.learn(total_timesteps=TIMESTEPS)
        close_tb_writer(wolf_model)
        vill_env.env.opponent_policy = wolf_model      # 好人对手 ← 最新狼

        print(f"\n>>> Round {r}/{ROUNDS}: Train Villagers <<<")
        vill_model.learn(total_timesteps=TIMESTEPS)
        close_tb_writer(vill_model)
        wolf_env.env.opponent_policy = vill_model      # 狼对手 ← 最新好人

    # ===== 保存模型 =====
    wolf_model.save("ppo_team_wolves_new")
    vill_model.save("ppo_team_villagers_new")
    print("=== 训练完成，模型已保存为 ppo_team_wolves_new.zip / ppo_team_villagers_new.zip ===")
