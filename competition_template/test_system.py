#!/usr/bin/env python3
"""
测试狼人杀比赛平台的基本功能
"""

import os
import sys
import tempfile
import subprocess
import importlib.util

def test_imports():
    """测试基本导入"""
    print("🔍 测试基本导入...")
    try:
        import numpy as np
        from werewolf_env.werewolf_env import WerewolfEnv, Role, TalkType
        from agents.base_agent import BaseAgent
        print("✅ 基本导入成功")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_evaluation_script():
    """测试评测脚本"""
    print("\n🔍 测试评测脚本...")
    try:
        # 创建一个简单的测试智能体
        test_code = '''
import numpy as np
from werewolf_env.werewolf_env import WerewolfEnv, Role, TalkType
from agents.base_agent import BaseAgent

class UserAgent(BaseAgent):
    def __init__(self, agent_id: int, num_agents: int, role: Role):
        super().__init__(agent_id, num_agents, role)
        self.role = role
        
    def act(self, env):
        if env.stage == "talk":
            return np.array([0, TalkType.CLAIM_GOOD, env.N])
        elif env.stage == "vote":
            return np.array([env.N])
        elif env.stage == "night":
            return np.array([env.N])
        return np.array([env.N])
'''
        
        # 保存测试代码
        test_file = 'agents_user/test_agent.py'
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_code)
        
        # 运行评测
        result = subprocess.run(
            ['python', 'evaluate_submission.py', 'agents_user.test_agent.UserAgent', '--games', '10'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # 清理测试文件
        os.remove(test_file)
        
        if result.returncode == 0:
            print("✅ 评测脚本运行成功")
            print(f"输出: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ 评测脚本运行失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_flask_app():
    """测试Flask应用"""
    print("\n🔍 测试Flask应用...")
    try:
        # 测试模板文件是否存在
        if not os.path.exists('templates/index.html'):
            print("❌ 模板文件不存在")
            return False
        
        # 测试Flask应用是否可以导入
        spec = importlib.util.spec_from_file_location("app", "app.py")
        if spec is None or spec.loader is None:
            print("❌ 无法加载app.py文件")
            return False
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        
        print("✅ Flask应用导入成功")
        return True
        
    except Exception as e:
        print(f"❌ Flask应用测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 狼人杀比赛平台系统测试")
    print("=" * 50)
    
    tests = [
        ("基本导入", test_imports),
        ("评测脚本", test_evaluation_script),
        ("Flask应用", test_flask_app)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 测试: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} 测试失败")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统可以正常运行")
        print("\n🚀 启动命令:")
        print("   python run_competition.py")
        print("   或")
        print("   python app.py")
    else:
        print("⚠️  部分测试失败，请检查系统配置")
        sys.exit(1)

if __name__ == '__main__':
    main() 