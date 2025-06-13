# -*- coding: utf-8 -*-
"""
eval_villagers_vs_beliefwolves.py
────────────────────────────────────────────────────────────
好人阵营：训练好的 PPO        (ppo_team_villagers_new.zip)
狼人阵营：BeliefAgent         (agents/belief_agent.py)
随机洗牌角色顺序，每局重新初始化各自的内部状态
"""

import os, sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from werewolf_env.werewolf_env import WerewolfEnv, Role
from utils.role_assigner import RoleAssigner
from agents.belief_agent import BeliefAgent

# ---------- 常量：角色配比 & 跑多少局 ----------
NUM_WOLVES, NUM_VILLAGERS = 1, 2          # 环境自动再加 1 预言家
N_EPISODES                = 500

# ---------- 载入训练好的好人 PPO ----------
vill_model = PPO.load("ppo_team_villagers_new.zip")

# ---------- 工具：把 obs dict flatten → 1-D 向量 ----------
def flatten_obs(od: dict) -> np.ndarray:
    return np.concatenate(
        [
            od["role"].astype(np.float32).ravel(),
            od["alive_mask"].astype(np.float32).ravel(),
            od["talk_history"].astype(np.float32).ravel(),
        ]
    )

# ---------- 工具：把统一离散编号 → env 动作 ----------
def decode_action(phase: str, a: int, role: Role, N: int):
    TALK_SIZE  = 2 * 3 * (N + 1)
    PHASE_SIZE = N + 1

    if phase == "talk":
        a %= TALK_SIZE
        ph_flag, rem = divmod(a, 3 * (N + 1))
        tt, tgt      = divmod(rem, N + 1)
        return np.array([ph_flag, tt, tgt], dtype=np.int32)

    if phase == "vote":
        return a % PHASE_SIZE

    if phase == "night":
        if role in (Role.WOLF, Role.SEER):
            return a % PHASE_SIZE
        return 0  # 村民夜晚 noop

    return 0

# ---------- 单局对战 ----------
def play_one(seed=None, render=False):
    # 1) 随机洗牌角色
    roles = RoleAssigner.assign_roles(NUM_WOLVES, NUM_VILLAGERS)
    env   = WerewolfEnv(roles, talk_history_len=20, max_nights=10, seed=seed)
    N     = len(roles)

    # 2) 构造 BeliefAgent（只给狼人）
    wolf_pids = [i for i, r in enumerate(roles) if r == Role.WOLF]
    vill_pids = [i for i, r in enumerate(roles) if r != Role.WOLF]
    belief_agents = {str(pid): BeliefAgent(pid, N, Role.WOLF) for pid in wolf_pids}
    for agent in belief_agents.values():
        env.add_agent(agent)

    obs_dict, _ = env.reset()
    done = False
    last_obs = obs_dict

    while not done:
        phase = env.stage
        full_act = {}

        # 2-A 狼人动作
        for pid_str, agent in belief_agents.items():
            full_act[pid_str] = agent.act(env)

        # 2-B 好人动作（由 PPO 一口气输出）
        vill_flat = np.concatenate(
            [flatten_obs(last_obs[str(pid)]) for pid in vill_pids], dtype=np.float32
        )
        vill_act_vec, _ = vill_model.predict(vill_flat, deterministic=True)
        for idx, pid in enumerate(vill_pids):
            a = int(vill_act_vec[idx])
            full_act[str(pid)] = decode_action(phase, a, roles[pid], N)

        # 3) 环境推进
        obs_dict, _, term_dict, trunc_dict, _ = env.step(full_act)
        last_obs = obs_dict
        done     = any(term_dict.values()) or any(trunc_dict.values())

    if render:
        env.render(n_events=50, god=True)

    return env._check_win()   # "WOLF" | "GOOD"

# ---------- 多局统计 ----------
if __name__ == "__main__":
    res = {"GOOD": 0, "WOLF": 0}
    for ep in range(1, N_EPISODES + 1):
        winner = play_one(seed=ep, render=(ep == 1 or ep == 2))
        res[winner] += 1
        print(f"Episode {ep:03d}  winner = {winner}")

    print("\n=== Final ===")
    print(f"好人胜率：{res['GOOD'] / N_EPISODES:.2%}")
    print(f"狼人胜率：{res['WOLF'] / N_EPISODES:.2%}")
