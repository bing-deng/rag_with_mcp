#!/bin/bash

# Milvus 快速启动脚本
# 此脚本将帮助您快速设置 Milvus 环境并运行示例

echo "🚀 Milvus 向量数据库快速启动脚本"
echo "=================================="

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    echo "   安装指南: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✅ Docker 已安装"

# 检查 Docker 是否运行
if ! docker info &> /dev/null; then
    echo "❌ Docker 未运行，请启动 Docker"
    exit 1
fi

echo "✅ Docker 正在运行"

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 未安装，请先安装 Python 3.7+"
    exit 1
fi

echo "✅ Python 3 已安装"

# 创建虚拟环境（可选）
read -p "是否创建 Python 虚拟环境？(y/n): " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv milvus_env
    source milvus_env/bin/activate
    echo "✅ 虚拟环境已创建并激活"
fi

# 安装 Python 依赖
echo "📦 安装 Python 依赖..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Python 依赖安装完成"

# 检查 Milvus 是否已经运行
if docker ps | grep -q milvus; then
    echo "✅ Milvus 已在运行"
else
    echo "🔧 启动 Milvus 服务器..."
    
    # 下载 Milvus standalone 启动脚本
    if [ ! -f "standalone_embed.sh" ]; then
        echo "📥 下载 Milvus 启动脚本..."
        curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
        chmod +x standalone_embed.sh
    fi
    
    # 启动 Milvus
    echo "🚀 启动 Milvus Standalone..."
    ./standalone_embed.sh start
    
    # 等待 Milvus 启动
    echo "⏳ 等待 Milvus 启动（30秒）..."
    sleep 30
fi

# 检查 Milvus 是否可用
echo "🔍 检查 Milvus 连接..."
python3 -c "
from pymilvus import connections
try:
    connections.connect('default', host='localhost', port='19530')
    print('✅ Milvus 连接成功')
    connections.disconnect('default')
except Exception as e:
    print(f'❌ Milvus 连接失败: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ 无法连接到 Milvus，请检查服务状态"
    exit 1
fi

echo ""
echo "🎉 环境设置完成！"
echo "===================="
echo ""
echo "现在您可以运行以下示例："
echo ""
echo "1. 基础示例：         python3 app.py"
echo "2. 文本搜索示例：     python3 text_search_example.py"
echo ""
echo "停止 Milvus 服务：    ./standalone_embed.sh stop"
echo "重启 Milvus 服务：    ./standalone_embed.sh restart"
echo ""

# 询问是否立即运行示例
read -p "是否立即运行基础示例？(y/n): " run_example
if [[ $run_example =~ ^[Yy]$ ]]; then
    echo "🏃 运行基础示例..."
    python3 app.py
fi

echo ""
echo "📚 更多信息请查看 README.md"
echo "�� 如果觉得有用，请给项目点个星！" 