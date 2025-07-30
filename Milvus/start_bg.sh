#!/bin/bash

echo "🚀 后台启动Web UI服务"

# 切换到脚本目录
cd "$(dirname "$0")"

# 停止现有进程
pkill -f "python.*web_app.py" 2>/dev/null

# 后台启动Web应用
nohup ./web_env/bin/python web_app.py > web_app.log 2>&1 &

sleep 2

# 检查是否启动成功
if curl -s http://localhost:5001/health > /dev/null 2>&1; then
    echo "✅ Web UI启动成功!"
    echo "📱 访问地址: http://localhost:5001"
    echo "📋 查看日志: tail -f web_app.log"
else
    echo "❌ Web UI启动失败，查看日志:"
    tail web_app.log
fi