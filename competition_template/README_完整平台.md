# 🐺 狼人杀智能体比赛平台 - 完整版

一个功能完整的狼人杀智能体比赛平台，支持用户注册、登录、代码提交、自动评测和排行榜功能。

## ✨ 功能特性

### 🔐 用户系统
- **用户注册/登录**: 支持用户名、邮箱和密码注册
- **个人资料**: 查看提交历史和成绩记录
- **会话管理**: 安全的用户会话管理

### 💻 代码提交与评测
- **在线代码编辑器**: 基于Monaco Editor的代码编辑环境
- **自动评测**: 提交代码后自动运行1000场游戏进行评测
- **多角色胜率**: 分别显示村民、狼人、预言家的胜率
- **代码历史**: 保存所有提交的代码版本

### 📊 数据管理
- **数据库存储**: 使用SQLite数据库存储用户数据和成绩
- **成绩记录**: 保存每次评测的详细结果
- **排行榜**: 实时更新的玩家排行榜

### 🎨 用户界面
- **现代化设计**: 响应式设计，支持移动端
- **直观导航**: 清晰的页面导航和用户反馈
- **实时反馈**: 评测进度和结果实时显示

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动平台

```bash
python run.py
```

### 3. 访问平台

打开浏览器访问: http://localhost:5000

### 4. 注册账号

- 点击"注册"按钮
- 填写用户名、邮箱和密码
- 完成注册后登录

### 5. 开始编程

- 在代码编辑器中编写你的智能体代码
- 点击"提交评测"按钮
- 等待评测结果

## 📁 项目结构

```
competition_template/
├── app.py                 # 主应用文件
├── models.py              # 数据库模型
├── forms.py               # 表单定义
├── run.py                 # 启动脚本
├── init_db.py             # 数据库初始化脚本
├── requirements.txt       # 依赖包列表
├── werewolf_competition.db # SQLite数据库文件
├── agents_user/           # 用户代码存储目录
├── templates/             # HTML模板
│   ├── index.html         # 主页面
│   ├── login.html         # 登录页面
│   ├── register.html      # 注册页面
│   ├── profile.html       # 个人资料页面
│   └── leaderboard.html   # 排行榜页面
├── werewolf_env/          # 狼人杀环境
└── agents/                # 官方智能体实现
```

## 🗄️ 数据库设计

### 用户表 (users)
- `id`: 用户ID
- `username`: 用户名
- `email`: 邮箱
- `password_hash`: 密码哈希
- `created_at`: 注册时间
- `last_login`: 最后登录时间

### 提交记录表 (submissions)
- `id`: 提交ID
- `user_id`: 用户ID
- `filename`: 文件名
- `code_content`: 代码内容
- `submitted_at`: 提交时间
- `is_active`: 是否为当前活跃版本

### 分数记录表 (scores)
- `id`: 分数ID
- `user_id`: 用户ID
- `submission_id`: 提交ID
- `overall_win_rate`: 总体胜率
- `villager_win_rate`: 村民胜率
- `wolf_win_rate`: 狼人胜率
- `seer_win_rate`: 预言家胜率
- `total_games`: 总游戏数
- `created_at`: 创建时间

## 🔧 配置说明

### 环境变量
- `SECRET_KEY`: Flask应用密钥（生产环境请修改）
- `SQLALCHEMY_DATABASE_URI`: 数据库连接字符串

### 评测配置
- 默认游戏数量: 1000场
- 评测超时时间: 120秒
- 支持的角色: 村民、狼人、预言家

## 🛠️ 开发指南

### 添加新功能

1. **添加新的数据库模型**:
   - 在 `models.py` 中定义新的模型类
   - 运行 `python init_db.py` 更新数据库

2. **添加新的页面路由**:
   - 在 `app.py` 中添加新的路由函数
   - 在 `templates/` 中创建对应的HTML模板

3. **修改评测逻辑**:
   - 编辑 `evaluate_submission.py` 文件
   - 调整游戏数量或评测参数

### 自定义样式

所有样式都在HTML模板的 `<style>` 标签中定义，可以根据需要修改CSS样式。

## 🔒 安全考虑

### 生产环境部署

1. **修改密钥**:
   ```python
   app.config['SECRET_KEY'] = 'your-production-secret-key'
   ```

2. **使用生产数据库**:
   ```python
   app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://...'
   ```

3. **启用HTTPS**:
   - 配置SSL证书
   - 使用反向代理（如Nginx）

4. **代码安全**:
   - 限制代码执行时间
   - 隔离代码执行环境
   - 监控系统资源使用

## 📈 性能优化

### 数据库优化
- 添加索引到常用查询字段
- 定期清理旧数据
- 使用数据库连接池

### 评测优化
- 并行处理多个评测任务
- 缓存评测结果
- 优化游戏环境性能

## 🐛 故障排除

### 常见问题

1. **数据库连接错误**:
   - 检查数据库文件权限
   - 确保SQLite支持已启用

2. **评测超时**:
   - 检查代码是否有无限循环
   - 增加评测超时时间

3. **导入错误**:
   - 确保所有依赖包已安装
   - 检查Python路径配置

### 日志查看

应用运行时会输出详细的日志信息，包括：
- 用户注册/登录
- 代码提交和评测
- 数据库操作
- 错误信息

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个平台！

### 开发流程

1. Fork项目
2. 创建功能分支
3. 提交代码更改
4. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**祝您在狼人杀智能体比赛中取得好成绩！** 🎉