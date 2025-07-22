from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import subprocess
import os
import tempfile
import uuid
import time
from datetime import datetime
import json
import sqlite3
import re

# 导入自定义模块
from models import db, User, Submission, Score, ModelFile
from forms import RegistrationForm, LoginForm, CodeSubmissionForm

app = Flask(__name__)

# 配置
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///werewolf_competition.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 初始化扩展
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '请先登录后再访问此页面'

# 配置上传文件夹 - 使用绝对路径确保在正确位置
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents_user')
MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

@login_manager.user_loader
def load_user(user_id):
    """加载用户"""
    return User.query.get(int(user_id))

@app.route('/')
def index():
    """主页面"""
    if current_user.is_authenticated:
        # 获取用户的最新分数
        latest_score = Score.query.filter_by(user_id=current_user.id).order_by(Score.created_at.desc()).first()
        return render_template('index.html', user=current_user, latest_score=latest_score)
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """用户注册"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data
        )
        user.set_password(form.password.data)
        
        db.session.add(user)
        db.session.commit()
        
        flash('注册成功！请登录', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """用户登录"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('用户名或密码错误', 'error')
    
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    """用户登出"""
    logout_user()
    flash('已成功登出', 'success')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    """用户个人资料页面"""
    # 获取用户的所有提交记录
    submissions = Submission.query.filter_by(user_id=current_user.id).order_by(Submission.submitted_at.desc()).all()
    
    # 获取用户的所有分数记录
    scores = Score.query.filter_by(user_id=current_user.id).order_by(Score.created_at.desc()).all()
    
    # 获取用户的最佳分数
    best_score = Score.query.filter_by(user_id=current_user.id).order_by(Score.overall_win_rate.desc()).first()
    
    return render_template('profile.html', 
                         user=current_user, 
                         submissions=submissions, 
                         scores=scores, 
                         best_score=best_score)

@app.route('/leaderboard')
def leaderboard():
    """排行榜页面"""
    # 获取所有用户的最佳分数 - 使用更简单的方法
    from sqlalchemy import func
    
    # 为每个用户找到最高总体胜率的记录
    subquery = db.session.query(
        Score.user_id,
        func.max(Score.overall_win_rate).label('max_overall_rate')
    ).group_by(Score.user_id).subquery()
    
    # 获取排行榜数据
    best_scores = db.session.query(
        User.username,
        Score.overall_win_rate,
        Score.villager_win_rate,
        Score.wolf_win_rate,
        Score.seer_win_rate,
        Score.created_at,
        Score.total_games
    ).join(Score).join(
        subquery,
        db.and_(
            Score.user_id == subquery.c.user_id,
            Score.overall_win_rate == subquery.c.max_overall_rate
        )
    ).order_by(Score.overall_win_rate.desc()).limit(50).all()
    
    return render_template('leaderboard.html', scores=best_scores)

@app.route('/submit_code', methods=['POST'])
@login_required
def submit_code():
    """提交代码接口"""
    try:
        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({'error': '请提供代码'}), 400
        
        user_code = data['code']
        
        # 生成唯一且合法的模块名和文件名
        def safe_module_name(user_id):
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            # 保证唯一id首字母是字母
            if not unique_id[0].isalpha():
                unique_id = 'a' + unique_id[1:]
            return f"user_{user_id}_{unique_id}"
        module_name = safe_module_name(current_user.id)
        filename = f"{module_name}.py"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # 保存用户代码到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(user_code)
        
        # 保存提交记录到数据库
        submission = Submission(
            user_id=current_user.id,
            filename=filename,
            code_content=user_code
        )
        db.session.add(submission)
        db.session.commit()
        
        # # 运行评测脚本
        # try:
        #     import_path = f"agents_user.{module_name}.UserAgent"
            
        #     # 获取当前脚本的绝对路径
        #     current_dir = os.path.dirname(os.path.abspath(__file__))
        #     evaluate_script = os.path.join(current_dir, 'evaluate_submission.py')

           
        # 运行评测脚本
        try:
            import_path = f"agents_user.{module_name}.UserAgent"
            
            # 获取当前脚本的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            evaluate_script = os.path.join(current_dir, 'evaluate_submission.py')
            
            # 设置环境变量确保编码一致
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['LANG'] = 'en_US.UTF-8'
            env['LC_ALL'] = 'en_US.UTF-8'
            
            # 使用更安全的subprocess调用
            result = subprocess.run(
                ['python', evaluate_script, import_path, '--games', '10000'],
                capture_output=True, 
                timeout=120,  # 2分钟超时
                cwd=current_dir,
                env=env
            )
            
            # 更健壮的编码处理
            def safe_decode(data):
                if not data:
                    return ""
                encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
                for encoding in encodings:
                    try:
                        return data.decode(encoding).strip()
                    except UnicodeDecodeError:
                        continue
                # 如果所有编码都失败，使用replace模式
                return data.decode('utf-8', errors='replace').strip()
            
            output = safe_decode(result.stdout)
            error = safe_decode(result.stderr)

            # result = subprocess.run(
            #     ['python', evaluate_script, import_path, '--games', '10000'],
            #     capture_output=True, 
            #     text=True, 
            #     timeout=120,  # 2分钟超时
            #     cwd=current_dir,
            #     encoding='utf-8'
            # )
            
            # output = result.stdout.strip() if result.stdout else ""
            # error = result.stderr.strip() if result.stderr else ""
            
            # 解析输出获取胜率
            win_rates = parse_evaluation_output(output)
            
            # 保存分数记录
            if win_rates:
                score = Score(
                    user_id=current_user.id,
                    submission_id=submission.id,
                    overall_win_rate=win_rates.get('overall', 0.0),
                    villager_win_rate=win_rates.get('villager', 0.0),
                    wolf_win_rate=win_rates.get('wolf', 0.0),
                    seer_win_rate=win_rates.get('seer', 0.0),
                    total_games=win_rates.get('total_games', 0),
                    villager_games=win_rates.get('villager_games', 0),
                    wolf_games=win_rates.get('wolf_games', 0),
                    seer_games=win_rates.get('seer_games', 0)
                )
                db.session.add(score)
                db.session.commit()
            
            # 保留用户代码文件，不删除
            # 文件将保存在 agents_user 目录中，方便后续查看和管理
            
            return jsonify({
                'success': True,
                'win_rates': win_rates,
                'output': output,
                'error': error,
                'filename': filename,
                'submission_id': submission.id
            })
            
        except subprocess.TimeoutExpired:
            return jsonify({'error': '评测超时，请检查代码是否有无限循环'}), 408
            
        except Exception as e:
            return jsonify({'error': f'评测失败: {str(e)}'}), 500
            
    except Exception as e:
                    return jsonify({'error': f'服务器错误: {str(e)}'}), 500

def parse_evaluation_output(output):
    """解析评测输出，提取胜率信息"""
    win_rates = {}
    
    try:
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            
            # 解析总体胜率
            if "总体胜率:" in line:
                try:
                    # 格式: "总体胜率: 123/456 = 26.97%"
                    parts = line.split("总体胜率:")[1].strip().split("=")
                    if len(parts) >= 2:
                        wins_total = parts[0].strip().split("/")
                        if len(wins_total) == 2:
                            wins = int(wins_total[0])
                            total = int(wins_total[1])
                            win_rates['overall'] = wins / total
                            win_rates['total_games'] = total
                except:
                    pass
            
            # 解析村民胜率
            elif "村民:" in line:
                try:
                    # 格式: "村民: 123/456 = 26.97%"
                    parts = line.split("村民:")[1].strip().split("=")
                    if len(parts) >= 2:
                        wins_total = parts[0].strip().split("/")
                        if len(wins_total) == 2:
                            wins = int(wins_total[0])
                            total = int(wins_total[1])
                            win_rates['villager'] = wins / total
                            win_rates['villager_games'] = total
                except:
                    pass
            
            # 解析狼人胜率
            elif "狼人:" in line:
                try:
                    # 格式: "狼人: 123/456 = 26.97%"
                    parts = line.split("狼人:")[1].strip().split("=")
                    if len(parts) >= 2:
                        wins_total = parts[0].strip().split("/")
                        if len(wins_total) == 2:
                            wins = int(wins_total[0])
                            total = int(wins_total[1])
                            win_rates['wolf'] = wins / total
                            win_rates['wolf_games'] = total
                except:
                    pass
            
            # 解析预言家胜率
            elif "预言家:" in line:
                try:
                    # 格式: "预言家: 123/456 = 26.97%"
                    parts = line.split("预言家:")[1].strip().split("=")
                    if len(parts) >= 2:
                        wins_total = parts[0].strip().split("/")
                        if len(wins_total) == 2:
                            wins = int(wins_total[0])
                            total = int(wins_total[1])
                            win_rates['seer'] = wins / total
                            win_rates['seer_games'] = total
                except:
                    pass
    
    except Exception as e:
        print(f"解析输出时出错: {e}")
    
    return win_rates

@app.route('/get_template')
def get_template():
    """获取代码模板"""
    template_code = '''import numpy as np
from werewolf_env.werewolf_env import WerewolfEnv, Role, TalkType

class UserAgent:
    """
    你的狼人杀智能体代码
    请实现以下方法：
    - __init__: 初始化
    - act: 主要行动方法
    
    注意：这个版本不继承BaseAgent，直接实现所需接口
    """
    
    def __init__(self, agent_id: int, num_agents: int, role: Role):
        """
        初始化智能体
        
        Args:
            agent_id: 自己的编号（0..N-1）
            num_agents: 总玩家数 N
            role: 角色类型
        """
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.role = role
        
        # 在这里添加你的初始化代码
        # 例如：记录游戏状态、策略参数等
        
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
            alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
            if alive_players:
                return np.array([alive_players[0]])  # 投第一个存活玩家
            return np.array([env.N])  # 默认不投票
            
        elif stage == "night":
            # 夜晚阶段
            # target: 目标玩家编号
            alive_players = [i for i in range(env.N) if env.alive[i] and i != self.agent_id]
            
            if self.role == Role.SEER:
                # 预言家查验
                if alive_players:
                    return np.array([alive_players[0]])  # 查验第一个存活玩家
                return np.array([env.N])
            elif self.role == Role.WOLF:
                # 狼人杀人
                if alive_players:
                    return np.array([alive_players[0]])  # 杀第一个存活玩家
                return np.array([env.N])
            else:
                # 村民夜晚无行动
                return np.array([env.N])
        
        return np.array([env.N])
'''
    return jsonify({'template': template_code})

@app.route('/get_user_code/<int:submission_id>')
@login_required
def get_user_code(submission_id):
    """获取用户提交的代码"""
    submission = Submission.query.get_or_404(submission_id)
    
    # 检查权限
    if submission.user_id != current_user.id:
        return jsonify({'error': '无权访问此代码'}), 403
    
    return jsonify({'code': submission.code_content})

@app.route('/list_user_files')
@login_required
def list_user_files():
    """列出当前用户的所有代码文件"""
    try:
        # 获取agents_user目录中的所有.py文件
        user_files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith('.py') and filename.startswith(f'user_{current_user.id}_'):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file_stat = os.stat(filepath)
                user_files.append({
                    'filename': filename,
                    'size': file_stat.st_size,
                    'modified': file_stat.st_mtime,
                    'module_name': filename[:-3]  # 去掉.py后缀
                })
        
        # 按修改时间排序，最新的在前
        user_files.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            'success': True,
            'files': user_files,
            'total_count': len(user_files)
        })
        
    except Exception as e:
        return jsonify({'error': f'获取文件列表失败: {str(e)}'}), 500

@app.route('/get_file_content/<filename>')
@login_required
def get_file_content(filename):
    """获取指定文件的内容"""
    try:
        # 安全检查：只能访问自己的文件
        if not filename.startswith(f'user_{current_user.id}_') or not filename.endswith('.py'):
            return jsonify({'error': '无权访问此文件'}), 403
        
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': '文件不存在'}), 404
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'content': content
        })
        
    except Exception as e:
        return jsonify({'error': f'读取文件失败: {str(e)}'}), 500

@app.route('/upload_model', methods=['POST'])
@login_required
def upload_model():
    """上传模型文件"""
    try:
        # 检查是否有文件上传
        if 'model_file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        file = request.files['model_file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 检查文件类型
        allowed_extensions = {'.pkl', '.pth', '.h5', '.pt', '.model', '.weights'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'不支持的文件类型: {file_ext}。支持的类型: {", ".join(allowed_extensions)}'}), 400
        
        # 检查文件大小 (限制为50MB)
        file.seek(0, 2)  # 移动到文件末尾
        file_size = file.tell()
        file.seek(0)  # 重置到文件开头
        
        if file_size > 50 * 1024 * 1024:  # 50MB
            return jsonify({'error': '文件大小超过50MB限制'}), 400
        
        # 生成唯一文件名
        unique_id = str(uuid.uuid4())[:8]
        if not unique_id[0].isalpha():
            unique_id = 'a' + unique_id[1:]
        
        safe_filename = f"user_{current_user.id}_{unique_id}{file_ext}"
        filepath = os.path.join(MODEL_FOLDER, safe_filename)
        
        # 保存文件
        file.save(filepath)
        
        # 保存到数据库
        model_record = ModelFile(
            user_id=current_user.id,
            filename=safe_filename,
            original_filename=file.filename,
            file_size=file_size,
            file_type=file_ext
        )
        db.session.add(model_record)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'filename': safe_filename,
            'original_filename': file.filename,
            'file_size': file_size,
            'message': '模型文件上传成功'
        })
        
    except Exception as e:
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

@app.route('/list_models')
@login_required
def list_models():
    """列出用户的模型文件"""
    try:
        models = ModelFile.query.filter_by(user_id=current_user.id).order_by(ModelFile.uploaded_at.desc()).all()
        
        model_list = []
        for model in models:
            filepath = os.path.join(MODEL_FOLDER, model.filename)
            exists = os.path.exists(filepath)
            model_list.append({
                'id': model.id,
                'filename': model.filename,
                'original_filename': model.original_filename,
                'file_size': model.file_size,
                'file_type': model.file_type,
                'uploaded_at': model.uploaded_at.isoformat(),
                'exists': exists
            })
        
        return jsonify({
            'success': True,
            'models': model_list
        })
        
    except Exception as e:
        return jsonify({'error': f'获取模型列表失败: {str(e)}'}), 500

@app.route('/delete_model/<int:model_id>', methods=['DELETE'])
@login_required
def delete_model(model_id):
    """删除模型文件"""
    try:
        model = ModelFile.query.get_or_404(model_id)
        
        # 检查权限
        if model.user_id != current_user.id:
            return jsonify({'error': '无权删除此文件'}), 403
        
        # 删除文件
        filepath = os.path.join(MODEL_FOLDER, model.filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # 删除数据库记录
        db.session.delete(model)
        db.session.commit()
        
        return jsonify({'success': True, 'message': '模型文件删除成功'})
        
    except Exception as e:
        return jsonify({'error': f'删除失败: {str(e)}'}), 500

# 创建数据库表
def create_tables():
    """创建数据库表"""
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    create_tables()
    app.run(debug=True, host='0.0.0.0', port=5000) 