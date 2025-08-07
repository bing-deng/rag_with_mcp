#  日本公司内部rag系统

### 架构

* 数据库： weavate
* embedding 使用 Amazon Bedrock的 cohere Embed Multilingual，包括出入库db 都使用cohere Embed Multilingual
* 使用LlamaIndex 做项目RAG的编排工作，聚焦结构化数据（PDF, DB, API）处理与索引，支持嵌入、摘要等

### 具体实现
* 公司pdf等资料 通过pdf 转唯 向量数据库，中间使用 Amazon Bedrock的Embed Multilingual 到 weaviate 数据库

### 可视化系统架构图（逻辑流程）

📄 文档（PDF/DB/API）      
   ↓
🔧 LlamaIndex - Loader + TextSplitter
   ↓
🔤 Amazon Bedrock (Cohere Embed Multilingual)
   ↓
🧠 Weaviate 向量数据库
   ↕（用户查询时）
❓Query → Embed（同样调用 Cohere）
   ↓
🔍 Weaviate 相似搜索（+ metadata filter）
   ↓
📚 LlamaIndex Query Engine
   ↓
🧠 Claude / GPT / Local LLM → 生成回答
   ↓
💬 展示 / API / 前端界面
