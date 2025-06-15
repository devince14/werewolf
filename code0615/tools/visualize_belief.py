
# tools/visualize_belief.py
from werewolf_env import Role   # ← 新增
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题
def plot_belief_over_time(history: list[np.ndarray],
                          x_labels: list[str],
                          agent_id: int,
                          belief_type: str = "wolf"):
    """
    history:   List of length T, 每项是长度 N 的 ndarray（某条信念分布在各人的概率）
    x_labels:  长度 T 的每个阶段文字标签
    agent_id:  被观察的 agent id
    belief_type: "wolf" 或 "seer"，决定我们在标题中标注哪条信念
    """
    # 转成 (T, N) array
    data = np.stack(history)  # shape (T, N)
    T, N = data.shape

    plt.figure(figsize=(8, 4))
    for i in range(N):
        plt.plot(range(T), data[:, i], label=f"P[{i}]")
    plt.xticks(range(T), x_labels, rotation=45, ha="right")
    plt.ylabel("Belief Probability")
    plt.xlabel("Game Phase")
    title = f"Agent {agent_id} belief of 'is {belief_type}' over time"
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()


def plot_all_beliefs(all_histories: dict[int, list[np.ndarray]],
                     x_labels: list[str],
                     roles: list[Role]):
    """
    all_histories: dict mapping agent_id -> belief-history (list of length-T arrays length-N)
    x_labels:      list of length T of phase labels
    roles:         list of Roles for each agent
    """
    N = len(roles)
    T = len(x_labels)

    # 自动选择网格行列
    n_cols = int(np.ceil(np.sqrt(N)))
    n_rows = int(np.ceil(N / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows),
                             sharex=True)
    # flatten 方便迭代
    axes = np.array(axes).reshape(-1)

    for aid in range(N):
        ax = axes[aid]
        hist = np.stack(all_histories[aid])  # shape (T, N_agents)
        belief_type = "seer" if roles[aid] == Role.WOLF else "wolf"
        for j in range(hist.shape[1]):
            ax.plot(range(T), hist[:, j], label=f"{j}")
        ax.set_ylabel(f"A{aid} P_{belief_type}", fontsize="small")
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize="x-small")
        ax.set_xticks(range(T))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize="x-small")

    # 隐藏多余子图
    for idx in range(N, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()