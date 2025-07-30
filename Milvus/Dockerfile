FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建templates目录（如果不存在）
RUN mkdir -p templates

# 暴露端口
EXPOSE 5000

# 设置环境变量
ENV FLASK_APP=web_app.py
ENV FLASK_ENV=production

# 启动命令
CMD ["python", "web_app.py"]