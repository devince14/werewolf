import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from werewolf_env import WerewolfEnv, Role, TalkType, PHASE_TALK, PHASE_LEGACY

# 1. 定义角色阵容：一只狼人、一个预言家、一个村民
roles = [Role.WOLF, Role.VILLAGER, Role.SEER]

# 2. 创建环境
env = WerewolfEnv(roles, talk_history_len=10, max_nights=5, seed=42)

# 3. 重置环境，获取初始观测
obs, info = env.reset()

# 4. 初始化终止标志
terminated = {agent: False for agent in obs.keys()}

# 5. 游戏主循环
round_idx = 0
while not all(terminated.values()):
    round_idx += 1

    # 随机为每个 agent 选一个动作
    action_dict = {}
    for agent in obs.keys():
        # 如果已经结束或该 agent 死了，选 noop
        if terminated[agent]:
            action_dict[agent] = env.noop_space.sample()
        else:
            space = env.action_space(agent)
            action_dict[agent] = space.sample()

    # 执行动作
    obs, rewards, terminated, truncated, info = env.step(action_dict)

    # 打印当前状态
    env.render(n_events=5, god=True)

# 6. 游戏结束
print("\nGame Over")
for agent, r in rewards.items():
    print(f"Player {agent} ({env.roles[int(agent)].name}): reward = {r}")
