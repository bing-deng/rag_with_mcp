# Milvus 向量数据库使用教程

## 📖 目录

1. [什么是 Milvus](#什么是-milvus)
2. [安装和环境设置](#安装和环境设置)
3. [核心概念](#核心概念)
4. [快速开始](#快速开始)
5. [详细代码解释](#详细代码解释)
6. [高级功能](#高级功能)
7. [性能优化](#性能优化)
8. [常见问题](#常见问题)
9. [最佳实践](#最佳实践)

## 什么是 Milvus

Milvus 是一个开源的向量数据库，专门用于存储、索引和搜索由深度神经网络和其他机器学习（ML）模型生成的大规模嵌入向量。它在以下场景中特别有用：

- **相似性搜索**：图像、音频、视频、文档的相似性搜索
- **推荐系统**：基于用户行为和内容特征的推荐
- **自然语言处理**：语义搜索、问答系统、聊天机器人
- **计算机视觉**：人脸识别、目标检测、图像分类
- **RAG 系统**：检索增强生成，为大语言模型提供外部知识

### 主要特性

- ✅ **高性能**：支持万亿级向量数据的秒级搜索
- ✅ **多种索引类型**：IVF、HNSW、DiskANN 等多种索引算法
- ✅ **云原生架构**：支持水平扩展和高可用性
- ✅ **多语言支持**：Python、Java、Go、Node.js、C++ 等
- ✅ **丰富的距离度量**：欧氏距离、内积、余弦相似度等

## 安装和环境设置

### 1. 系统要求

- **操作系统**：Linux、macOS、Windows
- **Python**：3.7 或更高版本
- **内存**：8GB 或更多（推荐 16GB+）
- **CPU**：x86_64 架构

### 2. 安装 Milvus 服务器

#### 方法一：使用 Docker（推荐）

```bash
# 下载并启动 Milvus Standalone
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start

# 或者使用 docker-compose
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d
```

#### 方法二：使用 Helm（Kubernetes）

```bash
helm repo add milvus https://milvus-io.github.io/milvus-helm/
helm repo update
helm install my-release milvus/milvus
```

#### 方法三：二进制安装

```bash
# 下载二进制文件
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-2.3.0-linux-amd64.tar.gz
tar -xzf milvus-2.3.0-linux-amd64.tar.gz
cd milvus-2.3.0-linux-amd64/bin
./milvus run standalone
```

### 3. 安装 Python 客户端

```bash
# 安装 PyMilvus
pip install pymilvus

# 如果需要安装其他依赖
pip install numpy pandas
```

### 4. 验证安装

启动 Milvus 服务后，可以通过以下方式验证：

```python
from pymilvus import connections
connections.connect("default", host="localhost", port="19530")
print("Milvus 连接成功！")
```

## 核心概念

### 1. 集合（Collection）

集合是 Milvus 中的基本数据组织单位，类似于关系数据库中的表。每个集合包含：

- **字段（Fields）**：定义数据结构
- **主键（Primary Key）**：唯一标识每条记录
- **向量字段**：存储嵌入向量
- **标量字段**：存储元数据

### 2. 字段类型

| 字段类型 | 描述 | 示例 |
|---------|------|------|
| `INT64` | 64 位整数 | 用作主键 ID |
| `FLOAT` | 32 位浮点数 | 分数、权重 |
| `DOUBLE` | 64 位浮点数 | 高精度数值 |
| `VARCHAR` | 变长字符串 | 文本描述 |
| `BOOL` | 布尔值 | 标记字段 |
| `FLOAT_VECTOR` | 浮点向量 | 嵌入向量 |
| `BINARY_VECTOR` | 二进制向量 | 压缩向量 |

### 3. 索引类型

| 索引类型 | 适用场景 | 特点 |
|---------|---------|------|
| `FLAT` | 小数据集，精确搜索 | 暴力搜索，100% 召回率 |
| `IVF_FLAT` | 中等数据集 | 聚类索引，平衡精度和速度 |
| `IVF_PQ` | 大数据集，内存限制 | 产品量化，低内存占用 |
| `HNSW` | 高精度要求 | 图索引，高召回率 |
| `DISKANN` | 超大数据集 | 磁盘索引，支持 TB 级数据 |

### 4. 距离度量

| 度量类型 | 计算公式 | 适用场景 |
|---------|---------|---------|
| `L2` | 欧氏距离 | 通用向量搜索 |
| `IP` | 内积 | 余弦相似度（归一化后） |
| `COSINE` | 余弦相似度 | 文本语义搜索 |
| `HAMMING` | 汉明距离 | 二进制向量 |
| `JACCARD` | 杰卡德距离 | 集合相似度 |

## 快速开始

### 1. 运行示例代码

确保 Milvus 服务正在运行，然后执行：

```bash
cd Milvus
python app.py
```

### 2. 预期输出

```
正在连接到 Milvus...
成功连接到 Milvus 服务器: localhost:19530
创建集合: demo_collection
集合 demo_collection 创建成功
生成 1000 条示例数据...
示例数据生成完成
正在插入数据...
插入数据完成，插入了 1000 条记录
正在创建索引...
索引创建完成
正在加载集合到内存...
集合加载完成

集合统计信息:
  集合名称: demo_collection
  记录数量: 1000
  集合描述: 这是一个演示集合
  字段信息:
    - id: DataType.INT64
    - text: DataType.VARCHAR
    - embedding: DataType.FLOAT_VECTOR

执行向量搜索，查询 3 个向量，返回前 5 个最相似的结果...

查询向量 1 的搜索结果:
  排名 1: ID=123, 距离=45.2341, 文本='这是第 123 条示例文本数据...'
  排名 2: ID=456, 距离=46.7823, 文本='这是第 456 条示例文本数据...'
  ...
```

## 详细代码解释

### 1. 连接管理

```python
def connect_to_milvus():
    """连接到 Milvus 服务器"""
    connections.connect("default", host=HOST, port=PORT)
```

**关键点：**
- `connections.connect()` 建立与 Milvus 的连接
- `"default"` 是连接的别名，可以管理多个连接
- 默认端口是 `19530`

### 2. 集合创建

```python
# 定义字段模式
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]

# 创建集合模式
schema = CollectionSchema(fields, "集合描述")
collection = Collection(COLLECTION_NAME, schema)
```

**关键点：**
- 每个集合必须有一个主键字段
- 向量字段必须指定维度 `dim`
- VARCHAR 字段需要指定最大长度 `max_length`
- `auto_id=False` 表示手动提供 ID

### 3. 数据插入

```python
def insert_data(collection, data):
    """插入数据，data 格式：[ids, texts, embeddings]"""
    insert_result = collection.insert(data)
    return insert_result
```

**数据格式要求：**
- 数据必须按字段顺序组织：`[主键, 字段1, 字段2, ...]`
- 向量数据必须是浮点数列表
- 批量插入比逐条插入效率更高

### 4. 索引创建

```python
index_params = {
    "metric_type": "L2",      # 距离度量
    "index_type": "IVF_FLAT", # 索引类型
    "params": {"nlist": 128}  # 索引参数
}
collection.create_index("embedding", index_params)
```

**索引参数说明：**
- `nlist`：IVF 索引的聚类中心数量
- `m`：PQ 索引的子空间数量
- `efConstruction`：HNSW 索引构建参数
- `M`：HNSW 索引的连接数

### 5. 向量搜索

```python
results = collection.search(
    query_vectors,           # 查询向量列表
    "embedding",            # 向量字段名
    search_params,          # 搜索参数
    limit=top_k,            # 返回结果数量
    output_fields=["id", "text"]  # 返回的字段
)
```

**搜索参数：**
- `nprobe`：IVF 索引搜索的聚类数量
- `ef`：HNSW 索引的搜索范围
- `search_k`：DiskANN 索引的搜索参数

## 高级功能

### 1. 分区管理

分区可以提高查询性能和数据管理效率：

```python
# 创建分区
collection.create_partition("partition_2023")
collection.create_partition("partition_2024")

# 在特定分区插入数据
collection.insert(data, partition_name="partition_2023")

# 在特定分区搜索
results = collection.search(
    query_vectors,
    "embedding",
    search_params,
    partition_names=["partition_2023"]
)
```

### 2. 表达式查询

支持复杂的条件查询：

```python
# 数值范围查询
results = collection.query(
    expr="id >= 100 and id <= 200",
    output_fields=["id", "text"]
)

# 字符串匹配
results = collection.query(
    expr='text like "特定关键词%"',
    output_fields=["id", "text"]
)

# 复合条件
results = collection.query(
    expr="id in [1, 2, 3] and text != ''",
    output_fields=["id", "text"]
)
```

### 3. 混合搜索

结合向量搜索和标量过滤：

```python
# 先过滤再搜索
results = collection.search(
    query_vectors,
    "embedding",
    search_params,
    expr="id >= 100",  # 标量过滤条件
    limit=10
)
```

### 4. 批量操作

```python
# 批量删除
collection.delete("id in [1, 2, 3, 4, 5]")

# 批量更新（通过删除+插入）
collection.delete("id in [1, 2, 3]")
collection.insert(new_data)
```

### 5. 集合别名

```python
from pymilvus import utility

# 创建别名
utility.create_alias(collection_name="demo_collection", alias="my_vectors")

# 使用别名访问集合
collection = Collection("my_vectors")
```

## 性能优化

### 1. 索引选择指南

```python
# 小数据集（< 10万）- 使用 FLAT
index_params = {
    "metric_type": "L2",
    "index_type": "FLAT"
}

# 中等数据集（10万 - 100万）- 使用 IVF_FLAT
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}  # 一般设为 sqrt(n)
}

# 大数据集（> 100万）- 使用 IVF_PQ
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_PQ",
    "params": {
        "nlist": 2048,
        "m": 8,  # 向量维度应该能被 m 整除
        "nbits": 8
    }
}

# 高精度要求 - 使用 HNSW
index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {
        "M": 16,
        "efConstruction": 200
    }
}
```

### 2. 搜索参数调优

```python
# IVF 索引搜索参数
search_params = {
    "metric_type": "L2",
    "params": {
        "nprobe": 128  # 增加 nprobe 提高召回率，但降低速度
    }
}

# HNSW 索引搜索参数
search_params = {
    "metric_type": "L2",
    "params": {
        "ef": 200  # 增加 ef 提高召回率
    }
}
```

### 3. 内存和并发优化

```python
# 设置资源池
from pymilvus import Config
Config.set_normalize_metric(False)  # 禁用度量标准化以提高性能

# 批量插入优化
batch_size = 1000
for i in range(0, len(all_data), batch_size):
    batch_data = all_data[i:i+batch_size]
    collection.insert(batch_data)
    collection.flush()  # 强制写入磁盘
```

## 常见问题

### Q1: 连接失败怎么办？

**A1:**
```python
# 检查 Milvus 服务状态
docker ps | grep milvus

# 检查端口是否开放
telnet localhost 19530

# 使用完整连接参数
connections.connect(
    alias="default",
    host="localhost",
    port="19530",
    user="root",       # 如果启用了认证
    password="Milvus"  # 默认密码
)
```

### Q2: 向量维度不匹配错误

**A2:**
```python
# 确保所有向量维度一致
DIMENSION = 128
embeddings = [[random.random() for _ in range(DIMENSION)] for _ in range(1000)]

# 检查向量维度
for i, embedding in enumerate(embeddings):
    if len(embedding) != DIMENSION:
        print(f"向量 {i} 维度错误: {len(embedding)}")
```

### Q3: 内存不足错误

**A3:**
```python
# 减小批处理大小
batch_size = 100  # 从 1000 减少到 100

# 释放集合内存
collection.release()

# 使用分区减少内存使用
collection.load(partition_names=["partition_2024"])
```

### Q4: 搜索结果为空

**A4:**
```python
# 确保集合已加载
collection.load()

# 检查索引状态
print(collection.indexes)

# 检查数据是否存在
print(f"集合记录数: {collection.num_entities}")

# 使用较大的搜索范围
search_params = {"metric_type": "L2", "params": {"nprobe": 128}}
```

## 最佳实践

### 1. 数据建模

```python
# 良好的字段设计
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="timestamp", dtype=DataType.INT64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
```

### 2. 索引策略

```python
# 为不同用途选择合适的索引
def create_optimal_index(collection, data_size):
    if data_size < 100000:
        # 小数据集：精确搜索
        index_params = {"metric_type": "L2", "index_type": "FLAT"}
    elif data_size < 1000000:
        # 中等数据集：平衡性能
        index_params = {
            "metric_type": "L2", 
            "index_type": "IVF_FLAT",
            "params": {"nlist": int(np.sqrt(data_size))}
        }
    else:
        # 大数据集：内存优化
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_PQ",
            "params": {"nlist": 2048, "m": 8, "nbits": 8}
        }
    
    collection.create_index("embedding", index_params)
```

### 3. 错误处理

```python
import logging
from pymilvus import MilvusException

def robust_milvus_operation():
    try:
        # Milvus 操作
        connections.connect("default", host="localhost", port="19530")
        collection = Collection("demo_collection")
        results = collection.search(query_vectors, "embedding", search_params)
        
    except MilvusException as e:
        logging.error(f"Milvus 错误: {e}")
        # 重试逻辑
        time.sleep(1)
        # 重新连接或其他恢复操作
        
    except Exception as e:
        logging.error(f"未知错误: {e}")
        
    finally:
        # 清理资源
        if connections.has_connection("default"):
            connections.disconnect("default")
```

### 4. 监控和维护

```python
def monitor_collection(collection):
    """监控集合状态"""
    stats = {
        "name": collection.name,
        "num_entities": collection.num_entities,
        "num_partitions": len(collection.partitions),
        "indexes": [index.params for index in collection.indexes],
        "loaded": collection.has_index()
    }
    return stats

# 定期清理
def cleanup_old_data(collection, days=30):
    """清理旧数据"""
    cutoff_timestamp = int(time.time()) - (days * 24 * 3600)
    collection.delete(f"timestamp < {cutoff_timestamp}")
```

## 总结

这个教程涵盖了 Milvus 向量数据库的核心功能和使用方法。通过运行 `app.py` 示例代码，您可以快速上手 Milvus 的基本操作。

### 学习路径建议

1. **初学者**：先运行示例代码，理解基本概念
2. **进阶者**：尝试不同的索引类型和搜索参数
3. **高级用户**：探索分区、表达式查询和性能优化

### 相关资源

- [Milvus 官方文档](https://milvus.io/docs)
- [PyMilvus API 参考](https://milvus.io/api-reference/pymilvus/v2.3.x/About.md)
- [Milvus GitHub 仓库](https://github.com/milvus-io/milvus)
- [社区论坛](https://discuss.milvus.io/)

如果您在使用过程中遇到问题，可以查看 [常见问题](#常见问题) 部分或访问官方社区寻求帮助。 