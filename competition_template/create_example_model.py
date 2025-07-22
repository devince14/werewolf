#!/usr/bin/env python3
"""
创建示例模型文件
用于演示强化学习功能
"""

import os
import pickle
import numpy as np

# 尝试导入torch（用于类定义）
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def create_sklearn_model():
    """创建一个简单的sklearn模型示例"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # 创建模拟数据
        np.random.seed(42)
        n_samples = 1000
        n_features = 28
        
        # 生成随机特征
        X = np.random.rand(n_samples, n_features)
        
        # 生成随机标签（模拟行动）
        y = np.random.randint(0, 5, n_samples)
        
        # 训练模型
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # 保存模型
        model_path = os.path.join(os.path.dirname(__file__), 'example_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✅ sklearn模型已保存到: {model_path}")
        return model_path
        
    except ImportError:
        print("❌ 需要安装scikit-learn: pip install scikit-learn")
        return None

# 定义简单的神经网络（移到函数外部）
if TORCH_AVAILABLE:
    class SimpleNet(nn.Module):
        def __init__(self, input_size=28, hidden_size=64, output_size=5):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

def create_pytorch_model():
    """创建一个简单的PyTorch模型示例"""
    if not TORCH_AVAILABLE:
        print("❌ 需要安装PyTorch: pip install torch")
        return None
    
    try:
        # 创建模型
        model = SimpleNet()
        
        # 保存模型
        model_path = os.path.join(os.path.dirname(__file__), 'example_model.pth')
        torch.save(model, model_path)
        
        print(f"✅ PyTorch模型已保存到: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ 创建PyTorch模型失败: {e}")
        return None

def create_joblib_model():
    """创建一个简单的joblib模型示例"""
    try:
        from sklearn.linear_model import LogisticRegression
        import joblib
        
        # 创建模拟数据
        np.random.seed(42)
        n_samples = 1000
        n_features = 28
        
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 5, n_samples)
        
        # 训练模型
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        
        # 保存模型
        model_path = os.path.join(os.path.dirname(__file__), 'example_model.model')
        joblib.dump(model, model_path)
        
        print(f"✅ joblib模型已保存到: {model_path}")
        return model_path
        
    except ImportError:
        print("❌ 需要安装joblib: pip install joblib")
        return None

def main():
    """主函数"""
    print("🎯 创建示例模型文件...")
    print("=" * 50)
    
    models_created = []
    
    # 创建sklearn模型
    print("\n1. 创建sklearn模型...")
    sklearn_path = create_sklearn_model()
    if sklearn_path:
        models_created.append(sklearn_path)
    
    # 创建PyTorch模型
    print("\n2. 创建PyTorch模型...")
    pytorch_path = create_pytorch_model()
    if pytorch_path:
        models_created.append(pytorch_path)
    
    # 创建joblib模型
    print("\n3. 创建joblib模型...")
    joblib_path = create_joblib_model()
    if joblib_path:
        models_created.append(joblib_path)
    
    print("\n" + "=" * 50)
    print("📊 创建结果:")
    
    if models_created:
        print(f"✅ 成功创建了 {len(models_created)} 个示例模型:")
        for path in models_created:
            print(f"   - {os.path.basename(path)}")
        
        print("\n📝 使用说明:")
        print("1. 将这些模型文件上传到平台的'模型管理'页面")
        print("2. 使用强化学习模板编写代码")
        print("3. 提交评测，系统会自动加载您的模型")
        
    else:
        print("❌ 没有成功创建任何模型")
        print("请检查是否安装了必要的依赖包:")
        print("  pip install scikit-learn torch joblib")
    
    print("\n🎉 完成！")

if __name__ == "__main__":
    main()