# AI问答系统流程分析

## 文件和功能分布

### 1. **web_app.py** - Web API入口
- **功能**: HTTP请求处理，JSON解析
- **流程**: 接收问题 → 创建LLaMAQueryEngine → 调用rag_query()
- **当前问题**: 
  - 每次请求都创建新的engine实例
  - 重复加载模型和建立连接
  - 没有缓存机制

### 2. **llama_query.py** - 核心RAG引擎
- **功能**: 语言检测、向量检索、LLaMA生成
- **流程**: 
  ```
  rag_query() → 语言检测 → basic_search() → 构建prompt → generate_response()
  ```
- **当前问题**:
  - 语言检测逻辑简单（仅基于字符）
  - prompt构建没有考虑检索质量
  - 没有结果排序和过滤

### 3. **query_milvus.py** - 向量检索引擎
- **功能**: 文本向量化、Milvus搜索、结果处理
- **流程**:
  ```
  basic_search() → text_to_vector() → collection.search() → 结果格式化
  ```
- **当前问题**:
  - 向量模型每次都要重新加载
  - 搜索参数固定，没有动态调整
  - 缺少查询扩展和重排序

## 性能瓶颈分析

### 🐌 **主要瓶颈**

1. **模型重复加载** (最大瓶颈)
   - Sentence-Transformers每次查询都加载
   - Ollama连接每次都建立
   - 估计耗时: 2-5秒

2. **数据库连接管理**
   - Milvus连接频繁建立/断开
   - 没有连接池
   - 估计耗时: 0.5-1秒

3. **向量计算重复**
   - 相同问题的向量没有缓存
   - 估计耗时: 0.1-0.3秒

4. **检索策略单一**
   - 只使用基础COSINE搜索
   - 没有混合检索（dense + sparse）
   - 没有重排序机制

## 详细优化建议

### 🚀 **性能优化** (影响最大)

#### 1. 模型单例模式
```python
class ModelManager:
    _instance = None
    _sentence_model = None
    _ollama_client = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_models()
        return cls._instance
```

#### 2. 连接池管理
```python
class MilvusConnectionPool:
    _connections = {}
    
    @classmethod
    def get_connection(cls, collection_name):
        if collection_name not in cls._connections:
            cls._connections[collection_name] = MilvusQueryEngine(collection_name)
        return cls._connections[collection_name]
```

#### 3. 查询缓存
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_text_to_vector(text: str):
    return model.encode(text)
```

### 🎯 **检索质量优化**

#### 1. 查询扩展
```python
def expand_query(original_query, lang):
    # 添加同义词、相关词
    if lang == 'ja':
        synonyms = {"会社": ["企業", "法人", "組織"], 
                   "情報": ["データ", "詳細", "概要"]}
```

#### 2. 混合检索策略
```python
def hybrid_search(query, top_k=10):
    # Dense检索 (当前的向量搜索)
    dense_results = basic_search(query, top_k*2)
    
    # Sparse检索 (关键词匹配)
    sparse_results = keyword_search(query, top_k*2)
    
    # 结果融合和重排序
    return rerank_results(dense_results, sparse_results, query)
```

#### 3. 动态搜索参数
```python
def adaptive_search_params(query_type, collection_stats):
    if query_type == "factual":
        return {"ef": 200, "metric_type": "COSINE"}
    elif query_type == "semantic":
        return {"ef": 100, "metric_type": "IP"}
```

### 🧠 **生成质量优化**

#### 1. 检索结果评分
```python
def score_retrieval_results(results, query):
    for result in results:
        result['relevance_score'] = calculate_relevance(result, query)
        result['quality_score'] = calculate_content_quality(result)
    return sorted(results, key=lambda x: x['relevance_score'], reverse=True)
```

#### 2. 动态Prompt调整
```python
def build_adaptive_prompt(question, results, detected_lang):
    # 根据检索结果质量调整prompt
    if avg_confidence < 0.7:
        return build_conservative_prompt(question, results, detected_lang)
    else:
        return build_detailed_prompt(question, results, detected_lang)
```

#### 3. 多轮对话支持
```python
class ConversationManager:
    def __init__(self):
        self.history = []
    
    def build_context_aware_prompt(self, current_question, history):
        # 考虑对话历史的prompt构建
```

## 实施优先级

### 🔥 **高优先级** (立即实施)
1. **模型单例模式** - 减少70%加载时间
2. **连接池管理** - 减少50%连接时间  
3. **基础缓存** - 减少重复计算

### 🔶 **中优先级** (1-2天内)
4. **查询扩展** - 提升检索准确率
5. **结果重排序** - 提升回答质量
6. **动态参数** - 适应不同查询类型

### 🔷 **低优先级** (长期规划)
7. **混合检索** - 复杂实现但效果好
8. **多轮对话** - 需要会话管理
9. **A/B测试框架** - 评估优化效果

## 预期性能提升

- **响应时间**: 从5-8秒降低到1-2秒 (75%提升)
- **准确率**: 从当前水平提升20-30%
- **并发处理**: 从单线程提升到支持10+并发
- **资源使用**: CPU和内存使用减少50%