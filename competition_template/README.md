
# 多模态决策实验室 狼人杀 AI “双周杯”比赛

该目录提供了简化的狼人杀环境以及基线 `BeliefAgent`。选手需在 `agents_user/` 编写自己的智能体，并使用评测脚本验证。

在当前版本v2.0中，我们提供了以下功能：

1. 环境：提供了简化的狼人杀环境，包括角色分配、阶段切换、游戏结束判定等。
2. 智能体：提供了一个基于信念的智能体 `BeliefAgent`，包含了验人、狼人和村民的策略。
3. 评测脚本：包含了一个用于评估智能体性能的脚本 `evaluate_submission.py`。评测环境为：
   * 10000 轮游戏，每轮最多 10 天。
   * 每个智能体有 1 个预言家(SEER)，1 个狼人(WOLF)，3 个村民(VILLAGE)。
   * **重要更新：选手现在会随机分配到狼人、预言家或村民角色！需要在代码中实现三种角色的策略。**


## 安装依赖

1. 使用 **Python 3.10+**。
2. 安装依赖包：
````markdown
   ```bash
   pip install -r requirements.txt
````

## 编写自定义 Agent

Agent 类需要实现 `__init__(agent_id, num_agents, role)` 和 `act(env)` 两个接口。`act` 接收 `WerewolfEnv` 实例，根据当前阶段返回合法动作。

**重要提醒：你的智能体需要能够处理三种不同的角色（狼人、预言家、村民）！**

示例代码如下：

```python
import numpy as np
from competition_template.werewolf_env.werewolf_env import Role, WerewolfEnv, TalkType

class MyAgent:
    def __init__(self, agent_id: int, num_agents: int, role: Role):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.role = role

    def act(self, env: WerewolfEnv):
        if env.stage == "talk":
            if self.role == Role.SEER:
                # 预言家策略：声称身份
                return np.array([1, TalkType.CLAIM_SEER, env.N])
            elif self.role == Role.WOLF:
                # 狼人策略：伪装好人
                return np.array([0, TalkType.CLAIM_GOOD, env.N])
            else:
                # 村民策略：声称好人
                return np.array([0, TalkType.CLAIM_GOOD, env.N])
        
        elif env.stage == "vote":
            # 投票策略（根据角色调整）
            alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
            if alive_players:
                return np.array([alive_players[0]])
            return np.array([env.N])
        
        elif env.stage == "night":
            if self.role == Role.SEER:
                # 预言家查验
                alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
                if alive_players:
                    return np.array([alive_players[0]])
                return np.array([env.N])
            elif self.role == Role.WOLF:
                # 狼人杀人
                alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
                if alive_players:
                    return np.array([alive_players[0]])
                return np.array([env.N])
            else:
                # 村民夜晚无行动
                return np.array([env.N])
        
        return np.array([env.N])
```

将文件放入 `agents_user/` 并确保类路径可导入，评测脚本会按类名实例化。

## 输入与输出格式

评测脚本会以如下方式创建你的 Agent：

```python
agent = MyAgent(agent_id, num_agents, role)
```

其中 `agent_id` 为智能体的编号(起始索引为0,默认值为0,表示)，`num_agents` 为游戏人数(默认值为5)，`role` 为 `Role` 枚举。随后在每个回合调用 `act(env)` 获取动作。

`act` 函数必须返回与当前阶段匹配的数据：

* **talk 阶段**：返回 `numpy.array([claim_seer, talk_type, target])`

  * `claim_seer`：`0` 表示不跳预言家，`1` 表示声明自己是预言家。
  * `talk_type`：`TalkType.CLAIM_GOOD`(0) 声称好人；`TalkType.ACCUSE`(1) 指控某人是狼；`TalkType.SUPPORT`(2) 支持某人。
  * `target`：目标玩家编号 `0..N-1`，或 `env.N` 表示无目标。
* **vote 阶段**：返回整数 `0..N-1` 表示投票对象，`env.N` 为弃票。
* **night 阶段**：

  * 狼人返回击杀目标 `0..N-1`，`env.N` 表示放弃。
  * 预言家返回查验目标 `0..N-1`，`env.N` 表示放弃。
  * 村民夜间无动作，可返回 `0` 或 `env.noop_space.sample()`。

确保返回值类型为 `int` 或 `numpy.ndarray`，并在合法范围内。

## 运行示例

执行下列命令可观看三名基线 `BeliefAgent` 的对局：

```bash
python -m competition_template.demo
```

脚本会运行一局并打印事件日志。

## 提交与评测

1. Fork 本仓库或在比赛系统下载模板，在 `competition_template/agents_user/` 下创建你的 Agent 文件。
2. 使用评测脚本本地验证：

   ```bash
   python -m competition_template.evaluate_submission competition_template.agents_user.my_agent.MyAgent --games 10
   ```
3. 在截止时间前提交 Pull Request 或上传压缩包，评测服务器会在干净环境中运行上述脚本。

### 评测说明

- **角色分配**：每局游戏中，你的智能体会随机分配到狼人、预言家或村民角色
- **胜率计算**：
  - 当你是狼人时，狼人获胜则该局获胜
  - 当你是预言家或村民时，好人获胜则该局获胜
  - 最终胜率 = 获胜局数 / 总局数
- **排名依据**：总体胜率（所有角色的平均胜率）

## 排行榜与奖励

评测结果会存入数据库并展示在网页排行榜。每两周根据最终排名奖励前三名：500 元、300 元、200 元。

