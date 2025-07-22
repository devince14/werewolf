#!/usr/bin/env python3
"""
PyCharm专用启动脚本
解决环境兼容性问题
"""

import os
import sys
import subprocess

def setup_environment():
    """设置PyCharm环境"""
    # 确保当前目录在Python路径中
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 设置PYTHONPATH环境变量
    os.environ['PYTHONPATH'] = current_dir
    
    print(f"🔧 设置工作目录: {current_dir}")
    print(f"🔧 设置Python路径: {sys.path[:3]}...")

def main():
    """主函数"""
    print("🐺 狼人杀智能体比赛平台 - PyCharm版本")
    print("=" * 50)
    
    # 检查是否在正确的目录
    if not os.path.exists('app_pycharm_fixed.py'):
        print("❌ 请先运行 python fix_pycharm_issue.py 创建修复版本")
        return
    
    # 设置环境
    setup_environment()
    
    # 启动修复版本的Flask应用
    print("🚀 启动修复版本的服务器...")
    print("📱 访问地址: http://localhost:5000")
    print("🛑 按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, 'app_pycharm_fixed.py'])
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")

if __name__ == '__main__':
    main()
