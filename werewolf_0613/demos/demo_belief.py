import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from werewolf_env import WerewolfEnv, Role
from agents.belief_agent import BeliefAgent
from tools.visualize_belief import plot_all_beliefs
from utils.role_assigner import RoleAssigner

def run_episode(roles, render=False):
    env = WerewolfEnv(roles)
    obs, _ = env.reset()
    N = len(roles)
    agents = {str(i): BeliefAgent(i, N, roles[i]) for i in range(N)}
    
    # 将agents添加到环境中
    for agent in agents.values():
        env.add_agent(agent)

    # 初始化每个 agent 的信念历史
    all_histories = {i: [] for i in range(N)}
    x_labels = []

    done = False
    while not done:
        # 阶段标签
        label = f"第{env.day+1}天："
        if env.stage == "night":  label += "夜晚"
        if env.stage == "talk":   label += "讨论"
        if env.stage == "vote":   label += "投票"
        x_labels.append(label)

        # 1) 所有 agent 同步更新并选动作
        actions = {aid: ag.act(env) for aid, ag in agents.items()}

        # 2) 环境推进
        obs, rewards, terms, truncs, info = env.step(actions)
        done = any(terms.values())

        # 3) 记录每个 agent 此刻的信念分布
        for i in range(N):
            bis = agents[str(i)].belief
            if roles[i] == Role.WOLF:
                all_histories[i].append(bis.P_seer.copy())
            else:
                all_histories[i].append(bis.P_wolf.copy())

    # 终局后再记录一次
    label = f"第{env.day}天：{env.stage}"
    x_labels.append(label)
    for i in range(N):
        bis = agents[str(i)].belief
        if roles[i] == Role.WOLF:
            all_histories[i].append(bis.P_seer.copy())
        else:
            all_histories[i].append(bis.P_wolf.copy())

    # 渲染回放
    if render:
        env.render(n_events=50, god=True)

    # 同时绘制所有 agent 的信念
    #plot_all_beliefs(all_histories, x_labels, roles)

    return env._check_win()

def main():
    # 使用RoleAssigner随机分配角色
    num_wolves = 2
    num_villagers = 3  # 不包括预言家
    n = 1  # 运行次游戏次数
    
    res = {"GOOD":0, "WOLF":0}
    for i in range(n):
        # 每次游戏重新随机分配角色
        roles = RoleAssigner.assign_roles(num_wolves, num_villagers)
        winner = run_episode(roles, render=(i==0))
        res[winner] += 1
        
    print(f"游戏总数: {n}")
    print(f"好人胜率：{res['GOOD']/n:.2%}")
    print(f"狼人胜率：{res['WOLF']/n:.2%}")

if __name__ == "__main__":
    main()
