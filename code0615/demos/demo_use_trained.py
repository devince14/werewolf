# -*- coding: utf-8 -*-
"""
demo_use_trained.py

加载并对战训练好的 “狼队” 与 “好人队” PPO 模型，
统计 100 局胜率示例。
"""
import os, sys
from typing import List, Tuple

import numpy as np
from stable_baselines3 import PPO

# ——————— 路径设置 ———————
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)                      # werewolf_env
sys.path.insert(0, os.path.join(REPO_DIR, "demos"))  # rl_team_selfplay

from werewolf_env.werewolf_env import WerewolfEnv, Role          # 最新环境
from rl_team_selfplay import TeamEnv                             # 之前定义的 TeamEnv

# ─────────────────────────────────────────────────────────────
#  一、对抗环境：CombinedEnv
# ─────────────────────────────────────────────────────────────
class CombinedEnv:
    """
    把最新 WerewolfEnv 封装成 “双阵营、模型对抗” 环境。
      • 夜晚阶段由各自模型决策；白天随机
      • obs_dict 直接交给 render 或调试使用
    """

    # ---------- 工具：flatten obs dict ----------
    @staticmethod
    def _flatten_obs_dict(od: dict) -> np.ndarray:
        return np.concatenate([
            od["role"].astype(np.float32).ravel(),         # (3,)
            od["alive_mask"].astype(np.float32).ravel(),   # (N,)
            od["talk_history"].astype(np.float32).ravel(), # (L×5,)
        ], dtype=np.float32)

    # ---------- 初始化 ----------
    def __init__(
        self,
        roles: List[Role],
        wolf_model: PPO,
        vill_model: PPO,
        max_nights: int = 5,
        seed: int | None = None,
    ):
        # 1) 原生并行环境
        self.raw = WerewolfEnv(
            roles=roles,
            talk_history_len=20,
            max_nights=max_nights,
            seed=seed,
        )

        # 2) 模型 & pid 划分
        self.wolf_model = wolf_model
        self.vill_model = vill_model
        self.wolf_pids  = [i for i, r in enumerate(roles) if r == Role.WOLF]
        self.vill_pids  = [i for i, r in enumerate(roles) if r != Role.WOLF]

        # 3) 缓存上一步 obs_dict
        self._last_obs = None

    # ---------- reset ----------
    def reset(self) -> Tuple[dict, dict]:
        obs_dict, info = self.raw.reset()
        self._last_obs = obs_dict
        return obs_dict, info

    # ---------- step ----------
    def step(self):
        phase = self.raw.stage
        full_act = {}

        # ========== 1) 狼队动作 ==========
        if phase == "night":
            wolf_obs = np.concatenate([
                self._flatten_obs_dict(self._last_obs[str(pid)])
                for pid in self.wolf_pids
            ], dtype=np.float32)
            wolf_act_vec, _ = self.wolf_model.predict(wolf_obs, deterministic=True)
            for idx, pid in enumerate(self.wolf_pids):
                legal_n = self.raw.action_space(str(pid)).n                  # N+1
                full_act[str(pid)] = int(wolf_act_vec[idx]) % legal_n
        else:  # talk / vote 阶段
            for pid in self.wolf_pids:
                full_act[str(pid)] = self.raw.action_space(str(pid)).sample()

        # ========== 2) 好人队动作 ==========
        if phase == "night":
            vill_obs = np.concatenate([
                self._flatten_obs_dict(self._last_obs[str(pid)])
                for pid in self.vill_pids
            ], dtype=np.float32)
            vill_act_vec, _ = self.vill_model.predict(vill_obs, deterministic=True)

            # 找到 Seer 的 pid（预言家夜验）
            seer_pid = next(i for i, r in enumerate(self.raw.roles) if r == Role.SEER)

            for idx, pid in enumerate(self.vill_pids):
                legal_n = self.raw.action_space(str(pid)).n
                if legal_n == 1:                               # 纯 noop（村民夜晚）
                    full_act[str(pid)] = 0
                    continue

                a = int(vill_act_vec[idx]) % legal_n           # 预测动作映射
                # 额外校正：Seer 必须验存活目标
                if pid == seer_pid and not (0 <= a < self.raw.N and self.raw.alive[a]):
                    alive_ids = [i for i, alive in enumerate(self.raw.alive)
                                 if alive and i != pid]
                    a = self.raw.random.choice(alive_ids) if alive_ids else pid
                full_act[str(pid)] = a
        else:  # talk / vote 随机
            for pid in self.vill_pids:
                full_act[str(pid)] = self.raw.action_space(str(pid)).sample()

        # ========== 3) 执行一步 ==========
        obs_dict, reward_dict, term_dict, trunc_dict, _ = self.raw.step(full_act)
        self._last_obs = obs_dict

        # — 汇总阵营奖励 —
        wolf_reward = sum(reward_dict[str(pid)] for pid in self.wolf_pids)
        vill_reward = sum(reward_dict[str(pid)] for pid in self.vill_pids)
        done = any(term_dict.values()) or any(trunc_dict.values())

        return obs_dict, (wolf_reward, vill_reward), done

# ─────────────────────────────────────────────────────────────
#  二、评估脚本
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) 载入模型（注意文件名保持与训练脚本一致）
    wolf_model = PPO.load("ppo_team_wolves_new.zip")          # ### NEW/CHANGED ###
    vill_model = PPO.load("ppo_team_villagers_new.zip")       # ### NEW/CHANGED ###

    # 2) 角色与环境保持训练时一致：2 狼 + 1 预言家 + 1 村民
    ROLES = [Role.WOLF, Role.VILLAGER, Role.SEER, Role.VILLAGER]  # ### NEW/CHANGED ###
    env = CombinedEnv(ROLES, wolf_model, vill_model, max_nights=5, seed=42)

    # 3) 对战 N_EPISODES 局
    N_EPISODES = 100
    win_count = {"WOLF": 0, "GOOD": 0}

    for ep in range(N_EPISODES):
        env.reset()
        done = False
        while not done:
            _, (wr, vr), done = env.step()
        env.raw.render(n_events=5, god=True)                   # 可注释

        winner = env.raw._check_win()  # "WOLF" | "GOOD"
        win_count[winner] += 1
        print(f"Episode {ep+1:03d}: winner = {winner}, rewards (W,V)=({wr:.1f},{vr:.1f})")

    print("\n=== Final win rates over", N_EPISODES, "episodes ===")
    print(win_count)
