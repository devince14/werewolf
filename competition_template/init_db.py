#!/usr/bin/env python3
"""
数据库初始化脚本
用于创建数据库表和初始数据
"""

from app import app, db
from models import User, Submission, Score

def init_database():
    """初始化数据库"""
    with app.app_context():
        # 创建所有表
        db.create_all()
        print("✅ 数据库表创建成功")
        
        # 检查是否已有管理员用户
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            # 创建管理员用户
            admin = User(
                username='admin',
                email='admin@werewolf.com'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("✅ 管理员用户创建成功 (用户名: admin, 密码: admin123)")
        else:
            print("ℹ️  管理员用户已存在")
        
        print("🎉 数据库初始化完成！")

if __name__ == '__main__':
    init_database() 