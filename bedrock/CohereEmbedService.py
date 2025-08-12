#!/usr/bin/env python3
"""
Cohere Embed Multilingual 专用服务类
基于实际测试结果优化的实现
"""

import boto3
import json
import numpy as np
from botocore.exceptions import ClientError
from typing import List, Dict, Any, Optional, Union
import time

class CohereEmbedService:
    """Cohere Embed Multilingual 专用服务类"""
    
    def __init__(self, region_name: str = 'ap-northeast-1'):
        """
        初始化Cohere嵌入服务
        
        Args:
            region_name: AWS区域名称
        """
        self.region_name = region_name
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name)
        
        # Cohere模型配置
        self.models = {
            'multilingual': 'cohere.embed-multilingual-v3',
            'english': 'cohere.embed-english-v3'
        }
        
        self.default_model = self.models['multilingual']
        
        # 基于测试结果的配置
        self.vector_dimension = 1024
        self.max_batch_size = 50  # 测试显示可以处理50个文本
        
    def _make_request(
        self, 
        texts: List[str], 
        input_type: str = "search_document",
        model: str = None,
        embedding_types: List[str] = None
    ) -> Optional[Dict]:
        """
        发送请求到Cohere模型
        
        Args:
            texts: 文本列表
            input_type: 输入类型
            model: 模型名称
            embedding_types: 嵌入类型
            
        Returns:
            API响应结果或None
        """
        if model is None:
            model = self.default_model
        elif model in self.models:
            model = self.models[model]
            
        if embedding_types is None:
            embedding_types = ["float"]
            
        try:
            body = {
                "texts": texts,
                "input_type": input_type,
                "embedding_types": embedding_types,
                "truncate": "END"
            }
            
            response = self.bedrock_runtime.invoke_model(
                modelId=model,
                contentType='application/json',
                accept='*/*',
                body=json.dumps(body)
            )
            
            result = json.loads(response['body'].read())
            return result
            
        except ClientError as e:
            print(f"Cohere API调用失败: {e}")
            return None
        except Exception as e:
            print(f"其他错误: {e}")
            return None
    
    def get_embeddings(
        self, 
        texts: Union[str, List[str]], 
        input_type: str = "search_document",
        model: str = None,
        batch_processing: bool = False
    ) -> List[List[float]]:
        """
        获取文本嵌入向量
        
        Args:
            texts: 单个文本或文本列表
            input_type: 输入类型 (search_document, search_query, classification, clustering)
            model: 模型名称 ('multilingual', 'english' 或完整模型ID)
            batch_processing: 是否强制批量处理
            
        Returns:
            嵌入向量列表
        """
        # 标准化输入
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return []
        
        # 基于测试结果：看起来Cohere模型可能需要逐个处理
        # 或者响应格式与预期不同
        embeddings = []
        
        if batch_processing and len(texts) <= self.max_batch_size:
            # 尝试批量处理
            result = self._make_request(texts, input_type, model)
            if result and 'embeddings' in result:
                embeddings = result['embeddings']
                
                # 检查返回的向量数量
                if len(embeddings) != len(texts):
                    print(f"⚠️ 批量处理异常: 期望{len(texts)}个向量，实际{len(embeddings)}个")
                    # 如果批量处理有问题，回退到逐个处理
                    return self._process_individually(texts, input_type, model)
                    
                return embeddings
            else:
                # 批量处理失败，回退到逐个处理
                return self._process_individually(texts, input_type, model)
        else:
            # 逐个处理（基于测试结果，这可能是更可靠的方式）
            return self._process_individually(texts, input_type, model)
    
    def _process_individually(
        self, 
        texts: List[str], 
        input_type: str,
        model: str
    ) -> List[List[float]]:
        """
        逐个处理文本（基于测试结果的备用方案）
        
        Args:
            texts: 文本列表
            input_type: 输入类型
            model: 模型名称
            
        Returns:
            嵌入向量列表
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            result = self._make_request([text], input_type, model)
            
            if result and 'embeddings' in result:
                embedding_list = result['embeddings']
                
                # 检查embeddings是否为列表且不为空
                if isinstance(embedding_list, list) and len(embedding_list) > 0:
                    # 检查第一个元素是否存在且不为None
                    if embedding_list[0] is not None:
                        embeddings.append(embedding_list[0])
                    else:
                        print(f"⚠️ 第{i+1}个文本的嵌入向量为空: {text[:30]}...")
                        embeddings.append(None)
                else:
                    print(f"⚠️ 第{i+1}个文本返回的embeddings格式异常: {text[:30]}...")
                    print(f"    embeddings内容: {embedding_list}")
                    embeddings.append(None)
            else:
                print(f"⚠️ 第{i+1}个文本请求失败: {text[:30]}...")
                if result:
                    print(f"    响应内容: {result}")
                embeddings.append(None)
            
            # 添加小延迟避免速率限制
            if i < len(texts) - 1:
                time.sleep(0.1)
        
        return embeddings
    
    def semantic_search(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 5,
        model: str = None
    ) -> List[Dict[str, Any]]:
        """
        语义搜索
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前K个结果
            model: 模型名称
            
        Returns:
            搜索结果列表
        """
        if not query or not documents:
            return []
        
        # 获取查询嵌入向量
        query_embeddings = self.get_embeddings(
            texts=[query],
            input_type="search_query",
            model=model
        )
        
        if not query_embeddings or query_embeddings[0] is None:
            print("❌ 查询嵌入向量生成失败")
            return []
        
        # 获取文档嵌入向量
        doc_embeddings = self.get_embeddings(
            texts=documents,
            input_type="search_document",
            model=model
        )
        
        if not doc_embeddings:
            print("❌ 文档嵌入向量生成失败")
            return []
        
        # 计算相似度
        query_vec = np.array(query_embeddings[0])
        similarities = []
        
        for i, doc_embedding in enumerate(doc_embeddings):
            if doc_embedding is not None:
                doc_vec = np.array(doc_embedding)
                
                # 余弦相似度
                similarity = np.dot(query_vec, doc_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
                )
                
                similarities.append({
                    'index': i,
                    'document': documents[i],
                    'similarity': float(similarity)
                })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def cross_lingual_search(
        self, 
        query: str, 
        multilingual_documents: List[Dict[str, str]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        跨语言搜索
        
        Args:
            query: 查询文本（任意语言）
            multilingual_documents: 多语言文档列表 [{"text": "内容", "language": "zh", "metadata": {...}}]
            top_k: 返回前K个结果
            
        Returns:
            跨语言搜索结果
        """
        documents = [doc["text"] for doc in multilingual_documents]
        
        # 使用多语言模型进行搜索
        search_results = self.semantic_search(
            query=query,
            documents=documents,
            top_k=top_k,
            model='multilingual'
        )
        
        # 添加语言和元数据信息
        for result in search_results:
            original_doc = multilingual_documents[result['index']]
            result['language'] = original_doc.get('language', 'unknown')
            result['metadata'] = original_doc.get('metadata', {})
        
        return search_results
    
    def classify_texts(
        self, 
        texts: List[str], 
        categories: List[str],
        model: str = None
    ) -> List[Dict[str, Any]]:
        """
        文本分类
        
        Args:
            texts: 待分类文本列表
            categories: 类别列表
            model: 模型名称
            
        Returns:
            分类结果列表
        """
        if not texts or not categories:
            return []
        
        # 获取类别嵌入向量
        category_embeddings = self.get_embeddings(
            texts=categories,
            input_type="classification",
            model=model
        )
        
        # 获取文本嵌入向量
        text_embeddings = self.get_embeddings(
            texts=texts,
            input_type="classification",
            model=model
        )
        
        if not category_embeddings or not text_embeddings:
            return []
        
        results = []
        
        for i, text_embedding in enumerate(text_embeddings):
            if text_embedding is None:
                results.append({
                    'text': texts[i],
                    'predicted_category': None,
                    'confidence': 0.0,
                    'all_scores': {}
                })
                continue
            
            text_vec = np.array(text_embedding)
            scores = {}
            best_category = None
            best_score = -1
            
            for j, category_embedding in enumerate(category_embeddings):
                if category_embedding is not None:
                    category_vec = np.array(category_embedding)
                    
                    # 计算余弦相似度
                    similarity = np.dot(text_vec, category_vec) / (
                        np.linalg.norm(text_vec) * np.linalg.norm(category_vec)
                    )
                    
                    scores[categories[j]] = float(similarity)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_category = categories[j]
            
            results.append({
                'text': texts[i],
                'predicted_category': best_category,
                'confidence': float(best_score),
                'all_scores': scores
            })
        
        return results
    
    def compute_similarity_matrix(
        self, 
        texts: List[str],
        input_type: str = "search_document",
        model: str = None
    ) -> np.ndarray:
        """
        计算文本间的相似度矩阵
        
        Args:
            texts: 文本列表
            input_type: 输入类型
            model: 模型名称
            
        Returns:
            相似度矩阵 (n x n)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.get_embeddings(
            texts=texts,
            input_type=input_type,
            model=model
        )
        
        if not embeddings or any(emb is None for emb in embeddings):
            print("❌ 部分嵌入向量生成失败")
            return np.array([])
        
        # 转换为numpy数组
        vectors = np.array(embeddings)
        
        # 计算相似度矩阵
        # 先归一化
        normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # 计算余弦相似度矩阵
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        
        return similarity_matrix
    
    def find_duplicates(
        self, 
        texts: List[str], 
        threshold: float = 0.95,
        model: str = None
    ) -> List[Dict[str, Any]]:
        """
        查找重复或近似重复的文本
        
        Args:
            texts: 文本列表
            threshold: 相似度阈值
            model: 模型名称
            
        Returns:
            重复文本组列表
        """
        similarity_matrix = self.compute_similarity_matrix(texts, model=model)
        
        if similarity_matrix.size == 0:
            return []
        
        duplicates = []
        processed = set()
        
        for i in range(len(texts)):
            if i in processed:
                continue
            
            # 找到与当前文本相似的所有文本
            similar_indices = []
            for j in range(i, len(texts)):
                if similarity_matrix[i, j] >= threshold:
                    similar_indices.append(j)
                    processed.add(j)
            
            if len(similar_indices) > 1:
                duplicates.append({
                    'group_id': len(duplicates),
                    'similar_texts': [
                        {
                            'index': idx,
                            'text': texts[idx],
                            'similarity_to_first': float(similarity_matrix[i, idx])
                        }
                        for idx in similar_indices
                    ],
                    'count': len(similar_indices)
                })
        
        return duplicates
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'available_models': self.models,
            'default_model': self.default_model,
            'vector_dimension': self.vector_dimension,
            'max_batch_size': self.max_batch_size,
            'supported_languages': '100+',
            'input_types': [
                'search_document',
                'search_query', 
                'classification',
                'clustering'
            ],
            'features': [
                'multilingual_support',
                'cross_lingual_search',
                'text_classification',
                'semantic_similarity',
                'duplicate_detection'
            ]
        }

# 使用示例和测试函数
def test_cohere_service():
    """测试Cohere服务的各项功能"""
    
    print("=== Cohere Embed Service 功能测试 ===")
    
    # 初始化服务
    cohere = CohereEmbedService(region_name='ap-northeast-1')
    
    # 1. 基本嵌入测试
    print("\n1. 基本嵌入向量测试...")
    texts = ["人工智能", "Artificial Intelligence", "人工知能"]
    embeddings = cohere.get_embeddings(texts)
    
    successful_embeddings = [emb for emb in embeddings if emb is not None]
    print(f"   📊 成功生成 {len(successful_embeddings)}/{len(texts)} 个嵌入向量")
    
    if successful_embeddings:
        print(f"   📐 向量维度: {len(successful_embeddings[0])}")
    
    # 2. 语义搜索测试
    print("\n2. 语义搜索测试...")
    query = "机器学习算法"
    documents = [
        "深度学习是机器学习的一个重要分支",
        "今天天气很好",
        "Machine learning algorithms are powerful",
        "自然语言处理技术发展迅速",
        "I like to eat pizza"
    ]
    
    search_results = cohere.semantic_search(query, documents, top_k=3)
    
    if search_results:
        print(f"   ✅ 搜索成功，找到 {len(search_results)} 个结果")
        for i, result in enumerate(search_results, 1):
            print(f"      {i}. {result['document'][:50]}... (相似度: {result['similarity']:.4f})")
    else:
        print("   ❌ 搜索失败")
    
    # 3. 跨语言搜索测试
    print("\n3. 跨语言搜索测试...")
    multilingual_docs = [
        {"text": "人工智能改变了世界", "language": "zh"},
        {"text": "AI is transforming the world", "language": "en"},
        {"text": "AIが世界を変えている", "language": "ja"},
        {"text": "今天吃什么好呢", "language": "zh"}
    ]
    
    cross_results = cohere.cross_lingual_search(
        query="What is artificial intelligence?",
        multilingual_documents=multilingual_docs,
        top_k=3
    )
    
    if cross_results:
        print(f"   ✅ 跨语言搜索成功")
        for result in cross_results:
            print(f"      [{result['language']}] {result['document']} (相似度: {result['similarity']:.4f})")
    else:
        print("   ❌ 跨语言搜索失败")
    
    # 4. 文本分类测试
    print("\n4. 文本分类测试...")
    classification_texts = [
        "这个产品质量很好，我很满意",
        "服务态度很差，不推荐",
        "价格还可以接受"
    ]
    categories = ["正面评价", "负面评价", "中性评价"]
    
    classification_results = cohere.classify_texts(classification_texts, categories)
    
    if classification_results:
        print("   ✅ 分类成功")
        for result in classification_results:
            if result['predicted_category']:
                print(f"      文本: {result['text']}")
                print(f"      分类: {result['predicted_category']} (置信度: {result['confidence']:.4f})")
    else:
        print("   ❌ 分类失败")
    
    # 5. 模型信息
    print("\n5. 模型信息...")
    model_info = cohere.get_model_info()
    print(f"   📋 可用模型: {list(model_info['available_models'].keys())}")
    print(f"   🌍 支持语言: {model_info['supported_languages']}")
    print(f"   📐 向量维度: {model_info['vector_dimension']}")

if __name__ == "__main__":
    test_cohere_service()