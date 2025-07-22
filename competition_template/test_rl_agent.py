#!/usr/bin/env python3
"""
测试强化学习智能体
"""

import os
import sys
import numpy as np
import pickle
from werewolf_env.werewolf_env import WerewolfEnv, Role, TalkType

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_model():
    """创建一个简单的测试模型"""
    # 创建一个简单的随机森林分类器作为示例
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # 创建一些模拟的训练数据
        np.random.seed(42)
        n_samples = 1000
        n_features = 28  # 与模板中的特征数量一致
        
        # 生成随机特征
        X = np.random.rand(n_samples, n_features)
        
        # 生成随机标签（模拟行动）
        y = np.random.randint(0, 5, n_samples)  # 0-4的行动
        
        # 训练模型
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # 保存模型
        model_path = os.path.join(os.path.dirname(__file__), 'test_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"测试模型已保存到: {model_path}")
        return model_path
        
    except ImportError:
        print("需要安装scikit-learn来创建测试模型")
        return None

def test_rl_agent():
    """测试强化学习智能体"""
    print("=== 测试强化学习智能体 ===")
    
    # 创建测试模型
    model_path = create_test_model()
    if not model_path:
        print("无法创建测试模型，跳过测试")
        return
    
    # 导入RL智能体模板
    try:
        from rl_agent_template import RLUserAgent
    except ImportError as e:
        print(f"无法导入RL智能体模板: {e}")
        return
    
    # 创建环境
    env = WerewolfEnv(N=5)
    
    # 测试不同角色
    roles = [Role.VILLAGER, Role.WOLF, Role.SEER]
    
    for role in roles:
        print(f"\n--- 测试角色: {role.name} ---")
        
        # 创建智能体
        agent = RLUserAgent(agent_id=0, num_agents=5, role=role)
        
        # 重置环境
        env.reset()
        
        # 模拟几个回合
        for round_num in range(3):
            print(f"  回合 {round_num + 1}:")
            
            # 获取当前状态
            current_state = env.get_observation(0)
            
            # 智能体行动
            action = agent.act(env)
            
            print(f"    状态: {current_state}")
            print(f"    行动: {action}")
            print(f"    模型加载状态: {agent.model_loaded}")
            
            # 模拟环境步进
            if env.stage == "talk":
                env.stage = "vote"
            elif env.stage == "vote":
                env.stage = "night"
            elif env.stage == "night":
                env.stage = "talk"
    
    print("\n=== 测试完成 ===")

def test_model_loading():
    """测试模型加载功能"""
    print("=== 测试模型加载功能 ===")
    
    # 创建测试模型
    model_path = create_test_model()
    if not model_path:
        return
    
    # 测试不同格式的模型加载
    test_formats = [
        ('.pkl', 'pickle'),
        ('.pth', 'torch'),
        ('.model', 'joblib')
    ]
    
    for ext, format_name in test_formats:
        print(f"\n测试 {format_name} 格式:")
        
        # 复制模型文件为不同格式
        new_path = model_path.replace('.pkl', ext)
        
        if ext == '.pth':
            try:
                import torch
                model = pickle.load(open(model_path, 'rb'))
                torch.save(model, new_path)
            except ImportError:
                print("  跳过: 需要安装torch")
                continue
        elif ext == '.model':
            try:
                import joblib
                model = pickle.load(open(model_path, 'rb'))
                joblib.dump(model, new_path)
            except ImportError:
                print("  跳过: 需要安装joblib")
                continue
        else:
            # 对于.pkl，直接复制
            import shutil
            shutil.copy2(model_path, new_path)
        
        print(f"  创建了 {new_path}")
        
        # 测试加载
        try:
            if ext == '.pkl':
                with open(new_path, 'rb') as f:
                    model = pickle.load(f)
            elif ext == '.pth':
                import torch
                model = torch.load(new_path, map_location='cpu')
            elif ext == '.model':
                import joblib
                model = joblib.load(new_path)
            
            print(f"  成功加载 {format_name} 模型")
            
        except Exception as e:
            print(f"  加载失败: {e}")
    
    print("\n=== 模型加载测试完成 ===")

if __name__ == "__main__":
    print("开始测试强化学习功能...")
    
    # 测试模型加载
    test_model_loading()
    
    # 测试RL智能体
    test_rl_agent()
    
    print("\n所有测试完成！")