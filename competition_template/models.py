from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """用户模型"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # 关联关系
    submissions = db.relationship('Submission', backref='user', lazy=True)
    scores = db.relationship('Score', backref='user', lazy=True)
    
    def set_password(self, password):
        """设置密码哈希"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """验证密码"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Submission(db.Model):
    """代码提交记录"""
    __tablename__ = 'submissions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    code_content = db.Column(db.Text, nullable=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)  # 是否为当前活跃版本
    
    def __repr__(self):
        return f'<Submission {self.filename} by {self.user_id}>'

class Score(db.Model):
    """分数记录"""
    __tablename__ = 'scores'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    submission_id = db.Column(db.Integer, db.ForeignKey('submissions.id'), nullable=False)
    
    # 各角色胜率
    villager_win_rate = db.Column(db.Float, default=0.0)
    wolf_win_rate = db.Column(db.Float, default=0.0)
    seer_win_rate = db.Column(db.Float, default=0.0)
    overall_win_rate = db.Column(db.Float, default=0.0)
    
    # 测试详情
    total_games = db.Column(db.Integer, default=0)
    villager_games = db.Column(db.Integer, default=0)
    wolf_games = db.Column(db.Integer, default=0)
    seer_games = db.Column(db.Integer, default=0)
    
    # 时间戳
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # 关联关系
    submission = db.relationship('Submission', backref='scores')
    
    def __repr__(self):
        return f'<Score {self.overall_win_rate:.2%} by {self.user_id}>'
    
    @property
    def formatted_overall_rate(self):
        """格式化总体胜率"""
        return f"{self.overall_win_rate:.2%}"
    
    @property
    def formatted_villager_rate(self):
        """格式化村民胜率"""
        return f"{self.villager_win_rate:.2%}"
    
    @property
    def formatted_wolf_rate(self):
        """格式化狼人胜率"""
        return f"{self.wolf_win_rate:.2%}"
    
    @property
    def formatted_seer_rate(self):
        """格式化预言家胜率"""
        return f"{self.seer_win_rate:.2%}"

class ModelFile(db.Model):
    """模型文件记录"""
    __tablename__ = 'model_files'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)  # 服务器文件名
    original_filename = db.Column(db.String(255), nullable=False)  # 原始文件名
    file_size = db.Column(db.Integer, nullable=False)  # 文件大小（字节）
    file_type = db.Column(db.String(10), nullable=False)  # 文件类型
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # 关联关系
    user = db.relationship('User', backref='model_files')
    
    def __repr__(self):
        return f'<ModelFile {self.original_filename} by {self.user_id}>'
    
    @property
    def formatted_size(self):
        """格式化文件大小"""
        if self.file_size < 1024:
            return f"{self.file_size} B"
        elif self.file_size < 1024 * 1024:
            return f"{self.file_size / 1024:.1f} KB"
        else:
            return f"{self.file_size / (1024 * 1024):.1f} MB"