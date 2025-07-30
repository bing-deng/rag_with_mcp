#!/usr/bin/env python3
"""
测试Web应用启动
"""

from flask import Flask, render_template, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({'status': 'ok', 'message': 'Web UI is running'})

@app.route('/api/test')
def test():
    """测试API"""
    return jsonify({'message': 'Test successful', 'port': 5001})

if __name__ == '__main__':
    print("🚀 测试Web服务启动")
    print("访问地址: http://localhost:5001")
    print("健康检查: http://localhost:5001/health")
    app.run(host='0.0.0.0', port=5001, debug=True)