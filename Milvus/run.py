#!/usr/bin/env python3
"""
简化的启动脚本
直接运行: python run.py
"""

import subprocess
import sys
import os

def main():
    # 确保在正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("🚀 启动智能查询系统 Web UI")
    print("=" * 50)
    
    # 检查虚拟环境
    venv_python = os.path.join(script_dir, 'web_env', 'bin', 'python')
    
    if not os.path.exists(venv_python):
        print("❌ 虚拟环境未找到，请先运行:")
        print("   ./start_web_ui.sh")
        sys.exit(1)
    
    # 启动Web应用
    print("🌐 启动Web服务...")
    print("访问地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务")
    print("-" * 50)
    
    try:
        subprocess.run([venv_python, 'web_app.py'])
    except KeyboardInterrupt:
        print("\n👋 Web服务已停止")

if __name__ == "__main__":
    main()