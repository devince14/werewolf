
````markdown
# 多模态决策实验室-梁舒 狼人杀 AI “双周杯”比赛

该目录提供了简化的狼人杀环境以及基线 `BeliefAgent`。选手需在 `agents_user/` 编写自己的智能体，并使用评测脚本验证。

## 安装依赖

1. 使用 **Python 3.10+**。
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
````

## 编写自定义 Agent

Agent 类需要实现 `__init__(agent_id, num_agents, role)` 和 `act(env)` 两个接口。`act` 接收 `WerewolfEnv` 实例，根据当前阶段返回合法动作。示例代码如下：

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
            # 不跳预言家，只声称好人
            return np.array([0, TalkType.CLAIM_GOOD, env.N])
        if env.stage == "vote":
            # 默认投票给 0 号玩家
            return 0
        if env.stage == "night" and self.role == Role.SEER:
            return 0  # 验证 0 号玩家
        return env.noop_space.sample()
```

将文件放入 `agents_user/` 并确保类路径可导入，评测脚本会按类名实例化。

## 输入与输出格式

评测脚本会以如下方式创建你的 Agent：

```python
agent = MyAgent(agent_id, num_agents, role)
```

其中 `agent_id` 为自己的编号，`num_agents` 为游戏人数，`role` 为 `Role` 枚举。随后在每个回合调用 `act(env)` 获取动作。

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
3. 在截止时间前提交 Pull Request 或上传压缩包，评测服务器会在干净环境中运行上述脚本统计狼方胜率。

## 排行榜与奖励

评测结果会存入 MySQL 数据库并展示在网页排行榜。示例查询程序位于 `scoreboard/Scoreboard.java`。每两周根据最终排名奖励前三名：500 元、300 元、200 元。

```

---

你可以直接复制保存为 `competition_template/README.md` 或贴到任何支持 Markdown 的平台如 GitHub、Notion、Typora 等。如果还需要转成 HTML 或 PDF，我也可以帮你。
```
