# 智能查询系统服务管理指南

## 🚀 服务重启方法

### 方法1: 完整重启 (推荐)
```bash
# 1. 停止服务
pkill -f "python.*web_app.py"

# 2. 等待2秒确保进程完全停止
sleep 2

# 3. 重新启动服务
nohup /Users/dengbingbing/Documents/startup-deng/rag_with_mcp/Milvus/web_env/bin/python /Users/dengbingbing/Documents/startup-deng/rag_with_mcp/Milvus/web_app.py > /Users/dengbingbing/Documents/startup-deng/rag_with_mcp/Milvus/web_app.log 2>&1 &

# 4. 检查服务状态
curl -s http://localhost:5001/health | head -3
```

### 方法2: 使用便捷脚本
```bash
cd /Users/dengbingbing/Documents/startup-deng/rag_with_mcp/Milvus
./start_bg.sh
```

### 方法3: 一键重启脚本
```bash
cd /Users/dengbingbing/Documents/startup-deng/rag_with_mcp/Milvus
python run.py
```

## 📊 服务状态检查

### 检查服务运行状态
```bash
# 检查进程
ps aux | grep web_app.py

# 健康检查API
curl http://localhost:5001/health

# 检查日志
tail -f /Users/dengbingbing/Documents/startup-deng/rag_with_mcp/Milvus/web_app.log
```

### 检查端口占用
```bash
# 检查5001端口
lsof -i :5001

# 检查Milvus端口
lsof -i :19530
```

## 🔧 服务管理命令

### 启动服务
```bash
# 前台启动 (调试用)
cd /Users/dengbingbing/Documents/startup-deng/rag_with_mcp/Milvus
./web_env/bin/python web_app.py

# 后台启动 (生产用)
nohup ./web_env/bin/python web_app.py > web_app.log 2>&1 &
```

### 停止服务
```bash
# 温和停止
pkill -f "python.*web_app.py"

# 强制停止
pkill -9 -f "python.*web_app.py"

# 停止所有Python进程 (谨慎使用)
killall python
```

### 重启服务
```bash
# 标准重启
pkill -f "python.*web_app.py" && sleep 2 && nohup ./web_env/bin/python web_app.py > web_app.log 2>&1 &

# 带预热的重启
pkill -f "python.*web_app.py" && sleep 2 && nohup ./web_env/bin/python web_app.py > web_app.log 2>&1 & && sleep 5 && curl -s http://localhost:5001/api/llama-chat -H "Content-Type: application/json" -d '{"question": "warmup", "collection_name": "kandenko_website"}' > /dev/null
```

## 📱 访问地址

- **Web UI**: http://localhost:5001
- **健康检查**: http://localhost:5001/health
- **API文档**: 启动时会在日志中显示

## 🐛 故障排除

### 常见问题和解决方案

#### 1. 端口被占用
```bash
# 找到占用进程
lsof -i :5001

# 释放端口
kill -9 <PID>
```

#### 2. 模型加载失败
```bash
# 检查虚拟环境
source /Users/dengbingbing/Documents/startup-deng/rag_with_mcp/Milvus/web_env/bin/activate
pip list | grep -E "(sentence|transformers|torch)"

# 重新安装依赖
pip install --upgrade sentence-transformers torch
```

#### 3. Milvus连接失败
```bash
# 检查Milvus服务
docker ps | grep milvus

# 重启Milvus
cd /Users/dengbingbing/Documents/startup-deng/rag_with_mcp/Milvus
docker-compose restart standalone
```

#### 4. Ollama服务不可用
```bash
# 检查Ollama
curl http://localhost:11434/api/tags

# 启动Ollama
ollama serve

# 拉取模型
ollama pull llama3.2:3b
```

## 📋 日志管理

### 查看日志
```bash
# 实时日志
tail -f web_app.log

# 最近100行
tail -100 web_app.log

# 搜索错误
grep -i error web_app.log

# 搜索特定查询
grep "RAG 查询" web_app.log
```

### 清理日志
```bash
# 清空日志
> web_app.log

# 备份并清空
mv web_app.log web_app.log.backup && touch web_app.log
```

## ⚡ 性能优化启动

### 预热启动 (推荐用于生产)
```bash
# 启动服务
nohup ./web_env/bin/python web_app.py > web_app.log 2>&1 &

# 等待服务启动
sleep 5

# 预热模型
curl -s -X POST http://localhost:5001/api/llama-chat \
  -H "Content-Type: application/json" \
  -d '{"question": "warmup test", "collection_name": "kandenko_website"}' > /dev/null

echo "✅ 服务已启动并预热完成"
```

## 🎯 当前服务状态

✅ **服务已重启完成**
- Web UI: http://localhost:5001  
- 健康状态: 正常
- Milvus连接: 正常 (6个集合)
- 模型管理器: 已初始化
- 优化功能: 已启用 (单例模式)

现在可以测试优化后的性能了！