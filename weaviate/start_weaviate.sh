#!/bin/bash

echo "=== 启动本地Weaviate RAG系统 ==="
echo ""

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker未运行，请先启动Docker"
    exit 1
fi

echo "🐳 启动Weaviate服务..."
docker-compose up -d

echo ""
echo "⏳ 等待服务启动..."
sleep 15  # 增加等待时间

# 检查服务状态
echo "📋 检查服务状态:"
docker-compose ps

echo ""
echo "🌐 服务访问地址:"
echo "  - Weaviate API: http://localhost:8180"
echo "  - Weaviate Console: http://localhost:8180/v1/meta"
echo "  - Text2Vec服务: http://localhost:8181"

echo ""
echo "📊 检查服务健康状态:"
echo "正在等待Weaviate启动..."
sleep 10

# 检查Weaviate是否就绪
if curl -f http://localhost:8180/v1/.well-known/ready > /dev/null 2>&1; then
    echo "✅ Weaviate已就绪!"
else
    echo "⚠️  Weaviate可能还未完全启动，请稍等..."
fi

echo ""
echo "✅ 服务已启动! 您可以运行以下命令测试:"
echo "  python weaviate/weaviate_client.py"
echo ""
echo "停止服务请运行: docker-compose down" 