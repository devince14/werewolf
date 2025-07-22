from flask import Flask, render_template, request, jsonify
import subprocess
import os
import tempfile
import uuid
import time
from datetime import datetime
import json

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'agents_user'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/submit_code', methods=['POST'])
def submit_code():
    """提交代码接口"""
    try:
        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({'error': '请提供代码'}), 400
        
        user_code = data['code']
        
        # 生成唯一文件名（使用时间戳+随机ID）
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        filename = f"user_{timestamp}_{unique_id}_agent.py"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # 保存用户代码
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(user_code)
        
        # 运行评测脚本
        try:
            # 构建导入路径 - 需要指定具体的类名
            import_path = f"agents_user.{filename[:-3]}.UserAgent"
            
            # 修复PyCharm subprocess环境问题
            env = os.environ.copy()
            env['PYTHONPATH'] = os.getcwd()
            
            # 运行评测（减少游戏数量以加快速度）
            result = subprocess.run(
                [sys.executable, 'evaluate_submission.py', import_path, '--games', '10000'],
                capture_output=True, 
                text=True, 
                timeout=60,  # 60秒超时
                cwd=os.getcwd(),
                env=env  # 使用修复的环境变量
            )
            
            # 解析输出（修复PyCharm兼容性）
            output = result.stdout.strip() if result.stdout else ""
            error = result.stderr.strip() if result.stderr else ""
            
            # 提取胜率（修复中文输出格式）
            win_rate = None
            if "总体胜率:" in output:
                try:
                    # 解析类似 "总体胜率: 3/10 = 30.00%" 的输出
                    rate_line = [line for line in output.split('\n') if '总体胜率:' in line][0]
                    rate_str = rate_line.split('=')[1].strip().replace('%', '')
                    win_rate = float(rate_str) / 100
                except:
                    pass
            
            # 清理临时文件
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify({
                'success': True,
                'win_rate': win_rate,
                'output': output,
                'error': error,
                'filename': filename
            })
            
        except subprocess.TimeoutExpired:
            # 清理临时文件
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({'error': '评测超时，请检查代码是否有无限循环'}), 408
            
        except Exception as e:
            # 清理临时文件
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({'error': f'评测失败: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/get_template')
def get_template():
    """获取代码模板"""
    template_code = '''import numpy as np
from werewolf_env.werewolf_env import WerewolfEnv, Role, TalkType
from agents.base_agent import BaseAgent

class UserAgent(BaseAgent):
    """
    你的狼人杀智能体代码
    请实现以下方法：
    - __init__: 初始化
    - act: 主要行动方法
    """
    
    def __init__(self, agent_id: int, num_agents: int, role: Role):
        super().__init__(agent_id, num_agents, role)
        self.role = role
        # 在这里添加你的初始化代码
        
    def act(self, env):
        """
        根据当前环境状态选择行动
        
        Args:
            env: 狼人杀环境对象
            
        Returns:
            numpy.array: 行动数组
                - talk阶段: [claim_seer, talk_type, target]
                - vote阶段: [target]
                - night阶段: [target]
        """
        # 获取当前阶段
        stage = env.stage
        
        if stage == "talk":
            # 发言阶段
            # claim_seer: 是否声称预言家 (0或1)
            # talk_type: 发言类型 (TalkType枚举)
            # target: 目标玩家编号
            return np.array([0, TalkType.CLAIM_GOOD, env.N])
            
        elif stage == "vote":
            # 投票阶段
            # target: 要投票的玩家编号
            return np.array([env.N])  # 默认不投票
            
        elif stage == "night":
            # 夜晚阶段
            # target: 目标玩家编号
            return np.array([env.N])  # 默认不行动
            
        return np.array([env.N])

# 示例：简单的随机策略
class RandomAgent(BaseAgent):
    def __init__(self, agent_id: int, num_agents: int, role: Role):
        super().__init__(agent_id, num_agents, role)
        self.role = role
        
    def act(self, env):
        import random
        
        if env.stage == "talk":
            # 随机发言
            claim_seer = random.choice([0, 1])
            talk_type = random.choice([TalkType.CLAIM_GOOD, TalkType.ACCUSE, TalkType.SUPPORT])
            target = random.choice([i for i in range(env.N) if env.alive[i] and i != self.agent_id])
            return np.array([claim_seer, talk_type, target])
            
        elif env.stage == "vote":
            # 随机投票
            alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
            if alive_players:
                target = random.choice(alive_players)
            else:
                target = env.N
            return np.array([target])
            
        elif env.stage == "night":
            # 夜晚行动
            if self.role == Role.SEER:
                # 预言家查验
                alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
                if alive_players:
                    target = random.choice(alive_players)
                else:
                    target = env.N
            elif self.role == Role.WOLF:
                # 狼人杀人
                alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
                if alive_players:
                    target = random.choice(alive_players)
                else:
                    target = env.N
            else:
                # 村民夜晚无行动
                target = env.N
            return np.array([target])
            
        return np.array([env.N])
'''
    return jsonify({'template': template_code})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 