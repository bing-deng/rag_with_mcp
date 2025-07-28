✅ 可用于 RAG 测试的公开数据集推荐
🔹 1. 维基百科文章（Wikipedia Dump）
内容类型：百科知识、结构化段落、可按主题组织

获取方式：

下载英文维基小型 dump: https://dumps.wikimedia.org/simplewiki/

或用 wikiextractor 提取纯文本

➡️ 非常适合 QA 问答测试（如“什么是人工智能？”）

🔹 2. 政府公开报告 / 白皮书
来源示例：

日本 METI 白皮书

美国能源部 DOE 文档库

欧盟文档库

➡️ 适合“问政策”、“总结规定”、“找数字”等场景

🔹 3. 论文数据集（ArXiv / PubMed）
使用 ArXiv.org 提供的 bulk 数据

或 CORD-19（COVID-19 文献数据集）：

地址：https://www.semanticscholar.org/cord19/download

➡️ 可测试“多篇文献摘要”、“科研对比问答”等高级 RAG 场景

🔹 4. 法律/合同/政策类数据
日本法令数据：https://www.digital.go.jp/policies/laws_regulations

美国联邦法律公开数据：https://www.govinfo.gov/

合同模板公开集：CUAD 合同分析数据集

➡️ 适合法律合规类企业场景，搭配 RAG 做问答/分析

🔹 5. 产品手册 / 开发文档 / API 文档
示例：

Python 官方文档：https://docs.python.org/3/download.html

OpenAPI 示例：https://github.com/OAI/OpenAPI-Specification

Notion / Slack / GitHub API 文档等

➡️ 测试“问开发手册”、“代码接入方式”、“参数意义”等问答场景

🔹 6. 金融数据/年报
日本 EDINET / TDNET 年报 PDF

美国 SEC 年报（10-K）：https://www.sec.gov/edgar/search/

Yahoo Finance API 抓取公司介绍

➡️ 测试“公司简介”、“营收变化”、“找关键数字”类 QA

📦 示例：CUAD 合同数据（法律文档）
bash
复制
编辑
git clone https://github.com/TheAtticusProject/cuad
cd cuad
# 提取 contract text，生成 chunks
每份合同是结构化的 JSON（带注释和条款），可直接嵌入后用 RAG 测试：

python
复制
编辑
{
  "contract_title": "Agreement XYZ",
  "text": "This Agreement is made and entered into...",
  "clauses": {
    "Termination": "The agreement may be terminated by either party..."
  }
}
🧪 如何快速测试？
可以按以下方式做 end-to-end：

下载 10~100 个文档（PDF、JSON、TXT 都行）

提取段落，切分成 chunks（每段 300~500 tokens）

使用本地嵌入模型生成向量（如 BGE）

存入 FAISS/Milvus

构造一个用户问题，检索最相似段落

把段落和问题拼成 Prompt，交给 LLaMA 本地模型

查看输出是否合理

