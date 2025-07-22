# 🚀 多模态决策实验室 狼人杀 AI “双周杯”比赛

## 📋 系统概述

这是一个类似LeetCode的狼人杀智能体比赛平台，支持：
- 在线编写智能体代码
- 实时评测和得分展示
- 多用户隔离（每个提交使用唯一文件名）
- 提交历史记录

## ⚡ 快速启动

### 1. 访问平台
打开浏览器访问：http://139.196.178.56:5000/

### 2. 注册账号

- 点击"注册"按钮
- 填写用户名、邮箱和密码
- 完成注册后登录

## 🎯 使用流程

### 第一步：加载代码模板
1. 打开网页后，点击"加载模板"按钮
2. 系统会自动加载基础代码模板到编辑器中

### 第二步：编写智能体代码
在代码编辑器中编写你的`UserAgent`类：

```python
class UserAgent:
    def __init__(self, agent_id: int, num_agents: int, role: Role):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.role = role
        
    def act(self, env):
        # 你的智能体逻辑
        if env.stage == "talk":
            return np.array([0, TalkType.CLAIM_GOOD, env.N])
        elif env.stage == "vote":
            return np.array([env.N])
        elif env.stage == "night":
            return np.array([env.N])
        return np.array([env.N])
```

### 第三步：提交评测
1. 点击"提交评测"按钮
2. 系统会运行10000场游戏进行评测
3. 在右侧查看评测结果和得分

## 📊 评测规则

- **游戏配置**：1狼人 + 3村民 + 1预言家，选手会随机分配到狼人、预言家或村民角色！需要在代码中实现三种角色的策略。
- **评测场数**：10000场
- **得分计算**：获胜的场次比例
- **超时限制**：60秒

## 🎮 游戏阶段说明

### 发言阶段 (`env.stage == "talk"`)
```python
return np.array([claim_seer, talk_type, target])
# claim_seer: 0或1，是否声称预言家
# talk_type: TalkType枚举值（CLAIM_GOOD, CLAIM_SEER, ACCUSE, SUPPORT）
# target: 目标玩家编号
```

### 投票阶段 (`env.stage == "vote"`)
```python
return np.array([target])
# target: 要投票的玩家编号
```

### 夜晚阶段 (`env.stage == "night"`)
```python
return np.array([target])
# target: 目标玩家编号
# - 预言家：查验目标
# - 狼人：杀死目标
# - 村民：无行动
```

## 🔧 开发技巧

### 1. 获取游戏信息
```python
def act(self, env):
    # 获取存活玩家
    alive_players = [i for i in range(env.N) if env.alive[i]]
    
    # 获取事件日志
    for event in env.event_log:
        if isinstance(event, dict):
            phase = event.get("phase")
            speaker = event.get("speaker")
            # 处理事件...
    
    # 获取当前阶段
    stage = env.stage
    
    # 获取当前天数
    day = env.day
```

### 2. 角色判断
```python
if self.role == Role.SEER:
    # 预言家逻辑
elif self.role == Role.WOLF:
    # 狼人逻辑
else:
    # 村民逻辑
```

### 3. 随机策略示例
```python
import random

def act(self, env):
    alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
    
    if env.stage == "talk":
        claim_seer = random.choice([0, 1])
        talk_type = random.choice([TalkType.CLAIM_GOOD, TalkType.ACCUSE, TalkType.SUPPORT])
        target = random.choice(alive_players) if alive_players else env.N
        return np.array([claim_seer, talk_type, target])
    
    elif env.stage == "vote":
        target = random.choice(alive_players) if alive_players else env.N
        return np.array([target])
    
    elif env.stage == "night":
        if self.role in [Role.SEER, Role.WOLF]:
            target = random.choice(alive_players) if alive_players else env.N
        else:
            target = env.N
        return np.array([target])
```

## 🐛 常见问题

### Q: 代码提交后没有结果？
A: 检查代码语法是否正确，确保类名为`UserAgent`

### Q: 评测超时？
A: 检查代码是否有无限循环，确保`act`方法能正常返回

### Q: 导入错误？
A: 确保使用了正确的导入路径，参考模板代码

### Q: 得分很低？
A: 这是正常的，狼人杀是一个复杂的博弈游戏，需要精心设计策略

## 📚 进阶资源

- 查看 `example_agent.py` 获取更多示例
- 阅读 `README_比赛平台.md` 了解详细文档
- 运行 `test_system.py` 检查系统状态

## 🎉 开始你的狼人杀之旅！

现在你已经了解了基本用法，开始编写你的智能体代码吧！记住：
- 多尝试不同的策略
- 分析游戏日志了解失败原因
- 参考其他优秀代码
- 享受编程和博弈的乐趣！

---

**提示**：使用 `Ctrl+Enter` 快捷键可以快速提交代码！ 
