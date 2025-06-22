# -*- coding: utf-8 -*-
"""
train_villagers_vs_beliefwolves.py
────────────────────────────────────────────────────────────
• 狼人阵营：BeliefAgent（固定策略，不学习）
• 好人阵营：PPO (预言家+村民×k) —— 学习 TALK / VOTE / NIGHT
"""

import os, sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, numpy as np
from typing import List, Dict
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import TensorBoardOutputFormat

from werewolf_env.werewolf_env import WerewolfEnv, Role
from utils.role_assigner import RoleAssigner
from agents.belief_agent import BeliefAgent

# =============  1. 轻量包装：VillTrainEnv  ============= #
class VillTrainEnv(gym.Env):
    """
    ① reset:  随机角色 -> WerewolfEnv -> 给每个狼人 pid 配 BeliefAgent
    ② step :  • 狼 = belief.act(env)
              • 好人 = PPO 动作 (联合观测)
    返回观测 = 好人全部 agent 的 obs flatten 拼接
    """
    metadata = {}

    def __init__(self,
                 num_wolves: int,
                 num_villagers: int,
                 opp_seed: int | None = None,
                 talk_history_len: int = 20,
                 max_nights: int = 15):
        super().__init__()
        self.nw  = num_wolves
        self.nv  = num_villagers
        self.talk_hist_len = talk_history_len
        self.max_nights    = max_nights
        self.rng = np.random.default_rng(opp_seed)

        roles0 = RoleAssigner.assign_roles(num_wolves, num_villagers)
        probe_env = WerewolfEnv(roles0, talk_history_len, max_nights)
        self.N = len(roles0)

        # obs / act 统一编码 (与前文一致)
        self.TALK_SIZE  = 2*3*(self.N+1)
        self.PHASE_SIZE = self.N+1
        self.ACTION_DIM = self.TALK_SIZE + 3*self.PHASE_SIZE + 1

        flat_sample, _ = probe_env.reset()
        vill_pids0 = [i for i,r in enumerate(roles0) if r != Role.WOLF]
        obs_dim = len(self._flatten(flat_sample[str(vill_pids0[0])])) * len(vill_pids0)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([self.ACTION_DIM]*len(vill_pids0))

        # for reset()
        self.raw: WerewolfEnv | None = None
        self.vill_pids: List[int] = []
        self.wolf_agents: Dict[str, BeliefAgent] = {}

    # -------- util --------
    @staticmethod
    def _flatten(od: dict) -> np.ndarray:
        return np.concatenate(
            [od["role"].astype(np.float32).ravel(),
             od["alive_mask"].astype(np.float32).ravel(),
             od["talk_history"].astype(np.float32).ravel()])

    def _decode(self, phase:str, a:int, role:Role):
        if phase=="talk":
            a %= self.TALK_SIZE
            ph, rem = divmod(a, 3*(self.N+1))
            tt, tgt = divmod(rem, self.N+1)
            return np.array([ph, tt, tgt], np.int32)
        if phase=="vote":
            return a % self.PHASE_SIZE
        if phase=="night":
            if role in (Role.WOLF, Role.SEER):
                return a % self.PHASE_SIZE
            return 0
        return 0

    # -------- Gym API --------
    def reset(self, *, seed=None, options=None):
        roles = RoleAssigner.assign_roles(self.nw, self.nv)
        self.raw = WerewolfEnv(roles, self.talk_hist_len, self.max_nights,
                               seed=self.rng.integers(1<<30))
        obs_dict, _ = self.raw.reset()

        # build wolf belief agents
        wolf_pids = [i for i,r in enumerate(roles) if r==Role.WOLF]
        self.wolf_agents = {str(pid): BeliefAgent(pid, self.N, Role.WOLF)
                            for pid in wolf_pids}
        for ag in self.wolf_agents.values():
            self.raw.add_agent(ag)

        self.vill_pids = [i for i,r in enumerate(roles) if r!=Role.WOLF]
        self.roles = roles
        self._last_obs = obs_dict
        return self._make_obs(obs_dict), {}

    def step(self, action):
        assert self.raw is not None
        phase = self.raw.stage
        full_act: Dict[str,int|np.ndarray] = {}

        # 1) 狼人动作
        for pid_str, ag in self.wolf_agents.items():
            full_act[pid_str] = ag.act(self.raw)

        # 2) 好人动作 (PPO 输出)
        for idx, pid in enumerate(self.vill_pids):
            a = int(action[idx]) % self.ACTION_DIM
            full_act[str(pid)] = self._decode(phase, a, self.roles[pid])

        # 3) env.step
        obs, rew, term, trunc, info = self.raw.step(full_act)
        self._last_obs = obs

        # ===== 4) 奖励计算 =====
        base_reward = sum(rew[str(pid)] for pid in self.vill_pids)
        shaped      = 0.0
        # ————— 塑形逻辑 —————
        phase = self.raw.stage
        seer_pid = next(i for i,r in enumerate(self.roles) if r == Role.SEER)

        # 4-1 Night
        if phase == "night" and len(self.raw.event_log) >= 1:
            ev_n = self.raw.event_log[-1]
            ev_s = self.raw.event_log[-2] if len(self.raw.event_log) >= 2 else {}
            # Seer 验人
            if ev_s.get("phase") == "seer_check":
                tgt_role = ev_s["role"]
                shaped += 0.5 if tgt_role == Role.WOLF.value else -0.1
            # 狼人击杀
            victim = ev_n.get("victim", -1)
            if victim == seer_pid:         shaped -= 0.5      # 预言家被砍
            elif victim >= 0 and self.roles[victim] == Role.WOLF:
                                            shaped += 0.1      # 狼误杀狼 → 好人加分

        # 4-2 Vote
        if phase == "vote" and len(self.raw.event_log) >= 1:
            
            ev = self.raw.event_log[-1]
            if ev.get("phase") == "vote":
                out = ev["out"]
                if out >= 0:
                    shaped += 0.2 if self.roles[out] == Role.WOLF else -0.2

        # 4-3 Talk（可选）
        if phase == "talk" and len(self.raw.event_log) >= 1:
            ev = self.raw.event_log[-1]
            if ev["phase"] == "talk" and ev["type"] == TalkType.ACCUSE:
                speaker, tgt = ev["speaker"], ev["target"]
                if self.roles[speaker] != Role.WOLF:           # 好人发言
                    shaped += 0.1 if self.roles[tgt] == Role.WOLF else -0.05

        # 4-4 终局
        done = any(term.values()) or any(trunc.values())
        if done and base_reward > 0:        shaped += 1.0

        # 4-5 归一化
        total_reward = base_reward + shaped  # 粗略上界

        return self._make_obs(obs), total_reward, done, False, info
    # -------- 观测拼接 --------
    def _make_obs(self, obs_dict):
        return np.concatenate(
            [self._flatten(obs_dict[str(pid)]) for pid in self.vill_pids],
            dtype=np.float32)

# =============  2. 辅助关闭 TensorBoard Writer  ============= #
def close_tb(model:PPO):
    for fmt in model.logger.output_formats:
        if isinstance(fmt, TensorBoardOutputFormat):
            try: fmt.writer.close()
            except Exception: pass

# =============  3. 训练主程序  ============= #
if __name__ == "__main__":
    NUM_WOLVES, NUM_VILLAGERS = 1, 2
    TIMESTEPS = 200_000
    ROUNDS    = 1

    env = Monitor(VillTrainEnv(NUM_WOLVES, NUM_VILLAGERS, opp_seed=0))
    # 如要继续在已有权重基础上微调，把下一行换成 PPO.load(...)
    # 2) PPO 超参 ----------------------
    POLICY_KW = dict(net_arch=[256, 256, 128],
                    activation_fn=torch.nn.ReLU)

    LINEAR_DECAY = lambda f: 3e-4 * f          # 迭代越晚 lr 越小

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/villagers_vs_beliefwolves",
        learning_rate=LINEAR_DECAY,
        n_steps=1024,
        gamma=0.95,
        gae_lambda=0.90,
        ent_coef=0.04,
        clip_range=0.2,
        policy_kwargs=POLICY_KW,
    )

    for r in range(1, ROUNDS+1):
        print(f"\n>>> Round {r}/{ROUNDS}: Train Villagers vs. Belief Wolves <<<")
        model.learn(total_timesteps=TIMESTEPS)
        close_tb(model)

    model.save("ppo_villagers_vs_beliefwolves")
    print("\n=== 训练完成，模型保存在 ppo_villagers_vs_beliefwolves.zip ===")
