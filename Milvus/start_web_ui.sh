#!/bin/bash

echo "🚀 启动智能查询系统 Web UI"
echo "=================================="

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

# 检查当前目录
if [ ! -f "web_app.py" ]; then
    echo "❌ 找不到web_app.py文件"
    exit 1
fi

# 创建虚拟环境（如果不存在）
if [ ! -d "web_env" ]; then
    echo "🔧 创建虚拟环境..."
    python3 -m venv web_env
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source web_env/bin/activate

# 安装依赖
echo "📦 安装依赖..."
pip install flask flask-cors pymilvus sentence-transformers beautifulsoup4 requests > /dev/null 2>&1

# 检查Milvus连接
echo "🔍 检查Milvus服务..."
if ! curl -s http://localhost:19530 > /dev/null; then
    echo "⚠️  Milvus服务未运行，请先启动:"
    echo "   docker-compose up -d"
    echo ""
fi

# 启动Web应用
echo "🌐 启动Web UI服务..."
echo "访问地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务"
echo ""

python web_app.py