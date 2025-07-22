# 狼人杀智能体比赛平台

这是一个基于Flask的狼人杀智能体比赛平台，允许用户在线编写和提交智能体代码进行评测。

## 功能特性

- 🎯 **在线代码编辑器**：使用Monaco Editor提供专业的代码编辑体验
- 🚀 **实时评测**：提交代码后立即进行评测，显示狼人胜率
- 📊 **结果展示**：清晰展示评测结果、输出信息和错误信息
- 📚 **提交历史**：记录最近的提交历史，方便查看
- 🔒 **多用户隔离**：每个提交使用唯一文件名，避免冲突

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务器

```bash
python app.py
```

### 3. 访问平台

打开浏览器访问：http://localhost:5000

## 使用说明

### 编写智能体代码

1. 点击"加载模板"按钮获取基础代码模板
2. 在代码编辑器中编写你的智能体逻辑
3. 确保你的类名为`UserAgent`

### 代码要求

你的智能体类必须：

1. 类名为`UserAgent`
2. 实现`__init__`和`act`方法
3. 不继承任何基类，直接实现接口

```python
class UserAgent:
    def __init__(self, agent_id: int, num_agents: int, role: Role):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.role = role
        
    def act(self, env):
        # 你的智能体逻辑
        # 返回numpy数组格式的行动
        pass
```

### 评测规则

- 游戏配置：1狼人 + 3村民 + 1预言家
- 评测场数：100场（可调整）
- 胜率计算：狼人获胜的场次比例
- 超时限制：60秒

### 行动格式

根据游戏阶段返回不同格式的行动：

- **发言阶段** (`env.stage == "talk"`)：
  ```python
  return np.array([claim_seer, talk_type, target])
  # claim_seer: 0或1，是否声称预言家
  # talk_type: TalkType枚举值
  # target: 目标玩家编号
  ```

- **投票阶段** (`env.stage == "vote"`)：
  ```python
  return np.array([target])
  # target: 要投票的玩家编号
  ```

- **夜晚阶段** (`env.stage == "night"`)：
  ```python
  return np.array([target])
  # target: 目标玩家编号（预言家查验/狼人杀人）
  ```

## 技术架构

### 前端
- **框架**：Flask + Jinja2模板
- **编辑器**：Monaco Editor（VS Code同款）
- **样式**：原生CSS + 响应式设计

### 后端
- **框架**：Flask
- **评测**：subprocess调用现有评测脚本
- **隔离**：时间戳+UUID生成唯一文件名

### 安全措施
- 代码执行超时限制（60秒）
- 临时文件自动清理
- 错误处理和异常捕获

## 文件结构

```
competition_template/
├── app.py                 # Flask主应用
├── templates/
│   └── index.html        # 前端页面
├── agents_user/          # 用户代码存储目录
├── evaluate_submission.py # 评测脚本
└── requirements.txt      # 依赖列表
```

## 开发说明

### 添加新功能

1. **用户系统**：可以添加用户注册、登录功能
2. **排行榜**：记录和展示用户得分排名
3. **代码分享**：允许用户分享和查看优秀代码
4. **详细统计**：提供更详细的游戏统计信息

### 性能优化

1. **异步评测**：使用Celery等任务队列
2. **缓存机制**：缓存评测结果
3. **数据库**：使用数据库存储用户数据和历史记录

## 故障排除

### 常见问题

1. **导入错误**：确保代码中使用了正确的导入路径
2. **语法错误**：检查Python语法是否正确
3. **超时错误**：检查代码是否有无限循环
4. **依赖缺失**：确保安装了所有必要的依赖包

### 调试模式

启动时添加调试信息：

```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 许可证

本项目遵循原项目的许可证。 