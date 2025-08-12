#!/bin/bash

echo "=== 智能文档问答系统启动脚本 ==="
echo ""

# 检查环境变量
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "❌ 请先设置AWS环境变量:"
    echo "export AWS_ACCESS_KEY_ID=your_key"
    echo "export AWS_SECRET_ACCESS_KEY=your_secret"
    echo "export AWS_DEFAULT_REGION=ap-northeast-1"
    echo ""
    echo "设置后请重新运行此脚本"
    exit 1
fi

echo "✅ AWS环境变量已设置"

# 检查PDF文件是否存在
PDF_PATH="../bedrock/pdf/high_takusoukun_web_manual_separate.pdf"
if [ ! -f "$PDF_PATH" ]; then
    echo "❌ PDF文件不存在: $PDF_PATH"
    echo "请确保PDF文件位于正确位置"
    exit 1
fi

echo "✅ PDF文件存在: $(basename "$PDF_PATH")"

# 检查Docker服务
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker未运行，请先启动Docker Desktop"
    exit 1
fi

echo "✅ Docker正在运行"

# 启动Weaviate（如果未运行）
cd ../
WEAVIATE_STATUS=$(docker-compose ps -q weaviate-server 2>/dev/null)
if [ -z "$WEAVIATE_STATUS" ] || [ "$(docker inspect -f '{{.State.Status}}' "$WEAVIATE_STATUS" 2>/dev/null)" != "running" ]; then
    echo "🐳 启动Weaviate服务..."
    docker-compose up -d
    echo "⏳ 等待Weaviate完全启动..."
    
    # 等待Weaviate就绪
    for i in {1..30}; do
        if curl -f http://localhost:8180/v1/.well-known/ready > /dev/null 2>&1; then
            echo "✅ Weaviate已就绪"
            break
        fi
        echo "等待中... ($i/30)"
        sleep 2
    done
    
    if [ $i -eq 30 ]; then
        echo "❌ Weaviate启动超时，请检查Docker状态"
        exit 1
    fi
else
    echo "✅ Weaviate已在运行"
fi

cd web_rag

# 清理旧的依赖冲突
echo "🧹 清理依赖冲突..."
pip uninstall grpcio -y > /dev/null 2>&1
pip uninstall python-dotenv -y > /dev/null 2>&1

# 安装Python依赖
echo "📦 安装Python依赖..."
pip install -r requirements.txt

# 检查关键依赖
echo "🔍 验证关键依赖..."
python -c "
import sys
try:
    import flask
    import weaviate  
    import boto3
    import PyPDF2
    import pdfplumber
    print('✅ 所有关键依赖都已正确安装')
except ImportError as e:
    print(f'❌ 依赖缺失: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ 依赖验证失败，请检查requirements.txt"
    exit 1
fi

# 创建必要的目录
mkdir -p templates static

echo ""
echo "🌐 启动Web应用..."
echo ""
echo "📋 系统信息:"
echo "  🔗 访问地址: http://localhost:5000"
echo "  📄 PDF文档: $(basename "$PDF_PATH")"
echo "  🗄️ 向量数据库: Weaviate (localhost:8180)"
echo "  🤖 AI模型: Claude 4 + Cohere"
echo ""
echo "🚀 系统功能:"
echo "  📄 智能PDF文档处理"
echo "  🔍 语义搜索和检索"
echo "  💬 基于上下文的智能问答"
echo "  📊 实时系统状态监控"
echo ""
echo "💡 使用提示:"
echo "  1. 打开浏览器访问 http://localhost:5000"
echo "  2. 点击'启动智能系统'进行初始化"
echo "  3. 等待系统处理PDF并创建向量数据库"
echo "  4. 开始提问，享受智能问答服务"
echo ""
echo "按 Ctrl+C 停止服务"
echo "=================================================="

# 启动Flask应用
python app.py 