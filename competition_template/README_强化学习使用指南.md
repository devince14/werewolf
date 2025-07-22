# 强化学习模型上传使用指南

## 🎯 概述

本平台未来将支持上传训练好的强化学习模型，让您的智能体使用训练好的策略进行游戏。
但是目前，您可以在源码中进行修改上传。
具体方法是：
在competition_template文件夹中，上传训练代码及模型，运行Python指令，进行评估得分。使用方法例如下：
python evaluate_submission.py agents_user.random_agent.UserAgent --games 1000

## 📁 支持的模型格式

- **`.pkl`** - Python pickle格式（sklearn、自定义模型等）
- **`.pth`** - PyTorch模型文件
- **`.pt`** - PyTorch模型文件（简化扩展名）
- **`.h5`** - Keras/TensorFlow模型文件
- **`.model`** - 自定义模型格式（使用joblib）
- **`.weights`** - 权重文件

## 🚀 使用步骤

### 1. 上传模型文件

1. 登录平台
2. 在个人资料页面找到"模型管理"部分
3. 点击"上传模型文件"
4. 选择您的模型文件（支持格式见上）
5. 文件大小限制：50MB

### 2. 使用强化学习模板

在代码编辑器中，使用以下模板：

```python
import numpy as np
import os
import pickle
import torch
import joblib
from werewolf_env.werewolf_env import WerewolfEnv, Role, TalkType

class UserAgent:
    def __init__(self, agent_id: int, num_agents: int, role: Role):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.role = role
        
        # 模型相关
        self.model = None
        self.model_loaded = False
        
        # 尝试加载模型
        self._load_model()
        
    def _load_model(self):
        """加载训练好的模型"""
        try:
            # 查找模型文件
            model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            if not os.path.exists(model_dir):
                print("模型目录不存在")
                return
            
            # 查找当前用户的模型文件
            user_models = []
            for filename in os.listdir(model_dir):
                if filename.startswith(f'user_{self.agent_id}_') and \
                   filename.endswith(('.pkl', '.pth', '.h5', '.pt', '.model', '.weights')):
                    user_models.append(filename)
            
            if not user_models:
                print("未找到模型文件")
                return
            
            # 使用最新的模型文件
            latest_model = sorted(user_models)[-1]
            model_path = os.path.join(model_dir, latest_model)
            
            print(f"尝试加载模型: {latest_model}")
            
            # 根据文件类型加载模型
            if latest_model.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                    
            elif latest_model.endswith('.pth') or latest_model.endswith('.pt'):
                # PyTorch模型
                self.model = torch.load(model_path, map_location='cpu')
                self.model.eval()
                
            elif latest_model.endswith('.h5'):
                # Keras模型
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                
            elif latest_model.endswith('.model'):
                # 自定义模型格式
                with open(model_path, 'rb') as f:
                    self.model = joblib.load(f)
                    
            elif latest_model.endswith('.weights'):
                # 权重文件
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            
            self.model_loaded = True
            print(f"模型加载成功: {latest_model}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None
            self.model_loaded = False
    
    def _extract_state_features(self, env):
        """提取环境状态特征"""
        features = []
        
        # 基本信息
        features.extend([
            self.agent_id,
            self.num_agents,
            int(self.role == Role.WOLF),
            int(self.role == Role.SEER),
            int(self.role == Role.VILLAGER),
            int(env.stage == "talk"),
            int(env.stage == "vote"),
            int(env.stage == "night")
        ])
        
        # 存活状态
        for i in range(env.N):
            features.append(int(env.alive[i]))
        
        # 游戏历史（简化版）
        if hasattr(env, 'talk_history') and env.talk_history:
            recent_talks = env.talk_history[-5:]  # 最近5轮发言
            for talk in recent_talks:
                features.extend([
                    talk.get('speaker', 0),
                    talk.get('talk_type', 0),
                    talk.get('target', env.N)
                ])
        else:
            # 填充空的历史
            features.extend([0, 0, env.N] * 5)
        
        return np.array(features, dtype=np.float32)
    
    def _get_action_from_model(self, state_features):
        """从模型获取行动"""
        if not self.model_loaded or self.model is None:
            return None
        
        try:
            # 根据模型类型进行推理
            if hasattr(self.model, 'predict'):
                # sklearn或其他predict接口
                action = self.model.predict([state_features])[0]
            elif hasattr(self.model, '__call__'):
                # PyTorch或其他可调用模型
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                with torch.no_grad():
                    action = self.model(torch.tensor(state_features).unsqueeze(0))
                    action = action.cpu().numpy()[0]
            else:
                # 自定义模型
                action = self.model(state_features)
            
            return action
            
        except Exception as e:
            print(f"模型推理失败: {e}")
            return None
    
    def _action_to_env_action(self, model_action, env):
        """将模型输出转换为环境行动"""
        if model_action is None:
            return self._default_action(env)
        
        try:
            # 根据当前阶段处理行动
            if env.stage == "talk":
                # talk阶段: [claim_seer, talk_type, target]
                claim_seer = int(model_action[0] > 0.5) if len(model_action) > 0 else 0
                talk_type = int(model_action[1]) if len(model_action) > 1 else TalkType.CLAIM_GOOD
                target = int(model_action[2]) if len(model_action) > 2 else env.N
                
                # 确保值在有效范围内
                talk_type = max(0, min(talk_type, len(TalkType) - 1))
                target = max(0, min(target, env.N))
                
                return np.array([claim_seer, talk_type, target])
                
            elif env.stage == "vote":
                # vote阶段: [target]
                target = int(model_action[0]) if len(model_action) > 0 else env.N
                target = max(0, min(target, env.N))
                return np.array([target])
                
            elif env.stage == "night":
                # night阶段: [target]
                target = int(model_action[0]) if len(model_action) > 0 else env.N
                target = max(0, min(target, env.N))
                return np.array([target])
                
            else:
                return self._default_action(env)
                
        except Exception as e:
            print(f"行动转换失败: {e}")
            return self._default_action(env)
    
    def _default_action(self, env):
        """默认行动策略（模型失败时的备用策略）"""
        if env.stage == "talk":
            return np.array([0, TalkType.CLAIM_GOOD, env.N])
        elif env.stage == "vote":
            alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
            if alive_players:
                return np.array([alive_players[0]])
            return np.array([env.N])
        elif env.stage == "night":
            alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
            if alive_players:
                return np.array([alive_players[0]])
            return np.array([env.N])
        else:
            return np.array([env.N])
    
    def act(self, env):
        """主要行动方法"""
        # 提取状态特征
        state_features = self._extract_state_features(env)
        
        # 从模型获取行动
        model_action = self._get_action_from_model(state_features)
        
        # 转换为环境行动
        action = self._action_to_env_action(model_action, env)
        
        return action
```

### 3. 提交评测

1. 确保已上传模型文件
2. 使用上述模板编写代码
3. 点击"提交评测"
4. 系统会自动加载您的模型并进行评测

## 🔧 模型要求

### 输入格式
模型应该接受一个特征向量作为输入，包含：
- 基本信息（8个特征）
- 存活状态（5个特征）
- 游戏历史（15个特征）
- 总计28个特征

### 输出格式
模型应该输出一个行动向量：
- **talk阶段**: [claim_seer, talk_type, target]
- **vote阶段**: [target]
- **night阶段**: [target]

## 📊 特征说明

### 基本信息 (8个特征)
- `agent_id`: 自己的编号
- `num_agents`: 总玩家数
- `is_wolf`: 是否为狼人
- `is_seer`: 是否为预言家
- `is_villager`: 是否为村民
- `is_talk_stage`: 是否为发言阶段
- `is_vote_stage`: 是否为投票阶段
- `is_night_stage`: 是否为夜晚阶段

### 存活状态 (5个特征)
- 每个玩家的存活状态（0或1）

### 游戏历史 (15个特征)
- 最近5轮发言的信息
- 每轮包含：发言者、发言类型、目标

## ⚠️ 注意事项

1. **模型兼容性**
   - 确保模型能在CPU上运行
   - PyTorch模型会自动转换为CPU模式
   - 大型模型可能影响评测速度

2. **文件大小**
   - 模型文件限制50MB
   - 建议压缩模型或使用量化技术

3. **错误处理**
   - 如果模型加载失败，会使用默认策略
   - 检查控制台输出了解错误信息

4. **依赖包**
   - 确保安装了相关依赖：torch, tensorflow, scikit-learn, joblib
   - 某些模型可能需要特定版本的依赖

## 🎮 示例

### 简单的sklearn模型
```python
# 训练时
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 保存模型
import pickle
with open('my_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### PyTorch模型
```python
# 训练时
import torch
model = MyNeuralNetwork()
# ... 训练过程 ...
torch.save(model, 'my_model.pth')
```

## 🚀 最佳实践

1. **模型优化**
   - 使用模型压缩技术减小文件大小
   - 确保模型推理速度快
   - 测试模型在不同场景下的表现

2. **特征工程**
   - 根据游戏特点设计合适的特征
   - 考虑添加更多游戏状态信息
   - 实验不同的特征组合

3. **训练策略**
   - 使用自对弈训练
   - 平衡探索和利用
   - 定期更新模型

## 🔍 测试功能

平台提供了测试脚本来验证强化学习功能：

```bash
python test_rl_agent.py
```

这个脚本会：
1. 创建一个简单的测试模型
2. 测试模型加载功能
3. 测试RL智能体的基本功能

## 📝 常见问题

### Q: 模型加载失败怎么办？
A: 检查以下几点：
- 模型文件格式是否正确
- 文件是否损坏
- 是否安装了相应的依赖包

### Q: 模型推理速度慢怎么办？
A: 可以尝试：
- 使用模型压缩技术
- 减少模型复杂度
- 使用更快的推理框架

### Q: 如何调试模型？
A: 在代码中添加print语句来查看：
- 模型加载状态
- 特征提取结果
- 模型输出结果

祝您训练出强大的狼人杀智能体！🎯 