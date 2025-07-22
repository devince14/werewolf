#!/usr/bin/env python3
"""
狼人杀智能体比赛平台启动脚本
"""

import os
import sys
import subprocess

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import flask
        import numpy
        print("✅ 依赖检查通过")
        return True
    except ImportError as e:
        print(f"❌ 依赖缺失: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def check_structure():
    """检查项目结构"""
    required_files = [
        'app.py',
        'templates/index.html',
        'evaluate_submission.py',
        'werewolf_env/__init__.py',
        'agents/__init__.py'
    ]
    
    # 检查当前目录是否在competition_template内
    if not os.path.exists('app.py') and os.path.exists('../app.py'):
        print("⚠️  检测到在上级目录运行，切换到competition_template目录...")
        os.chdir('..')
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少必要文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ 项目结构检查通过")
    return True

def create_agents_user_dir():
    """创建用户代码目录"""
    if not os.path.exists('agents_user'):
        os.makedirs('agents_user')
        print("✅ 创建 agents_user 目录")

def main():
    """主函数"""
    print("🐺 狼人杀智能体比赛平台")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查项目结构
    if not check_structure():
        sys.exit(1)
    
    # 创建必要目录
    create_agents_user_dir()
    
    print("\n🚀 启动服务器...")
    print("📱 访问地址: http://localhost:5000")
    print("🛑 按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    try:
        # 启动Flask应用
        subprocess.run([sys.executable, 'app.py'], cwd=os.getcwd())
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == '__main__':
    main() 