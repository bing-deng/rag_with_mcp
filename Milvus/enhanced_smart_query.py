#!/usr/bin/env python3
"""
Enhanced Smart Query System
Advanced AI training optimization with improved vector search, ranking, and relevance scoring
"""

import json
import math
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pymilvus import connections, utility, Collection
from query_milvus import MilvusQueryEngine
from llama_query import LLaMAQueryEngine

# Advanced text processing
import re
from collections import Counter
from dataclasses import dataclass

@dataclass
class QueryContext:
    """Enhanced query context with business intelligence"""
    query: str
    intent: str  # 'factual', 'analytical', 'navigational', 'transactional'
    domain: str  # 'business', 'technical', 'financial', 'general'
    importance_keywords: List[str]
    boost_content_types: List[str]
    temporal_preference: str  # 'recent', 'historical', 'any'

@dataclass
class SearchResult:
    """Enhanced search result with comprehensive scoring"""
    id: int
    content: str
    content_type: str
    url: str
    title: str
    word_count: int
    timestamp: int
    
    # Scoring components
    semantic_score: float
    keyword_score: float
    business_relevance_score: float
    content_type_score: float
    temporal_score: float
    combined_score: float
    
    # Context information
    matched_keywords: List[str]
    business_entities: List[str]

class EnhancedMilvusQueryEngine(MilvusQueryEngine):
    """Enhanced query engine with advanced ranking and optimization"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Business domain keywords for boosting
        self.business_keywords = {
            'financial': ['売上', '利益', '売上高', '営業利益', '純利益', '資本金', '負債', '資産', '株価', '配当'],
            'company_info': ['設立', '創立', '従業員', '本社', '支社', '事業所', '資本金', 'CEO', '代表取締役'],
            'business_activities': ['事業', 'サービス', '製品', '技術', 'ソリューション', 'プロジェクト', '建設', '工事'],
            'governance': ['コーポレートガバナンス', 'CSR', '持続可能性', 'サステナビリティ', 'IR', '株主総会'],
            'performance': ['業績', '実績', '成果', '達成', '向上', '改善', '効果', '効率']
        }
        
        # Content type importance weights
        self.content_type_weights = {
            'company_info': 1.5,
            'business_div': 1.4,
            'heading_h1': 1.3,
            'heading_h2': 1.2,
            'key_data': 1.4,
            'definition_list': 1.3,
            'paragraph': 1.0,
            'list': 1.1,
            'table': 1.2,
            'emphasis': 1.2,
            'quote': 1.1
        }
    
    def analyze_query_intent(self, query: str) -> QueryContext:
        """Advanced query intent analysis"""
        query_lower = query.lower()
        
        # Intent classification
        intent = 'general'
        if any(word in query_lower for word in ['何', 'どの', 'どう', 'なぜ', 'いつ', 'どこ']):
            intent = 'factual'
        elif any(word in query_lower for word in ['比較', '分析', '評価', '傾向', '推移']):
            intent = 'analytical'
        elif any(word in query_lower for word in ['探す', '見つける', 'ページ', 'サイト']):
            intent = 'navigational'
        elif any(word in query_lower for word in ['申込', '問い合わせ', '連絡', 'お問い合わせ']):
            intent = 'transactional'
        
        # Domain classification
        domain = 'general'
        financial_score = sum(1 for keyword in self.business_keywords['financial'] if keyword in query)
        company_score = sum(1 for keyword in self.business_keywords['company_info'] if keyword in query)
        business_score = sum(1 for keyword in self.business_keywords['business_activities'] if keyword in query)
        
        if financial_score > 0:
            domain = 'financial'
        elif company_score > 0:
            domain = 'company'
        elif business_score > 0:
            domain = 'business'
        
        # Extract important keywords
        importance_keywords = []
        for category, keywords in self.business_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    importance_keywords.append(keyword)
        
        # Determine content type boost based on domain
        boost_content_types = []
        if domain == 'financial':
            boost_content_types = ['table', 'key_data', 'definition_list']
        elif domain == 'company':
            boost_content_types = ['company_info', 'company_div', 'definition_list']
        elif domain == 'business':
            boost_content_types = ['business_div', 'heading_h1', 'heading_h2']
        
        # Temporal preference (simplified)
        temporal_preference = 'any'
        if any(word in query_lower for word in ['最新', '新しい', '現在', '今年']):
            temporal_preference = 'recent'
        elif any(word in query_lower for word in ['歴史', '過去', '以前', '創立']):
            temporal_preference = 'historical'
        
        return QueryContext(
            query=query,
            intent=intent,
            domain=domain,
            importance_keywords=importance_keywords,
            boost_content_types=boost_content_types,
            temporal_preference=temporal_preference
        )
    
    def calculate_keyword_score(self, content: str, query_context: QueryContext) -> Tuple[float, List[str]]:
        """Calculate keyword-based relevance score"""
        content_lower = content.lower()
        query_lower = query_context.query.lower()
        
        # Basic keyword matching
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        
        common_words = query_words.intersection(content_words)
        basic_score = len(common_words) / len(query_words) if query_words else 0
        
        # Business keyword boosting
        business_matches = []
        business_boost = 0
        for keyword in query_context.importance_keywords:
            if keyword in content:
                business_matches.append(keyword)
                business_boost += 0.2
        
        # Exact phrase matching
        phrase_boost = 0
        if len(query_context.query) > 5 and query_context.query in content:
            phrase_boost = 0.3
        
        total_score = min(basic_score + business_boost + phrase_boost, 1.0)
        matched_keywords = list(common_words) + business_matches
        
        return total_score, matched_keywords
    
    def calculate_business_relevance_score(self, content: str, content_type: str, query_context: QueryContext) -> float:
        """Calculate business context relevance score"""
        score = 0.0
        
        # Domain-specific boosting
        if query_context.domain == 'financial':
            financial_indicators = ['円', '億', '万', '%', '率', '年度', '売上', '利益']
            score += sum(0.1 for indicator in financial_indicators if indicator in content)
        
        elif query_context.domain == 'company':
            company_indicators = ['会社', '企業', '法人', '株式会社', '代表', '役員', '従業員']
            score += sum(0.1 for indicator in company_indicators if indicator in content)
        
        elif query_context.domain == 'business':
            business_indicators = ['事業', 'サービス', '技術', 'ソリューション', '顧客', 'お客様']
            score += sum(0.1 for indicator in business_indicators if indicator in content)
        
        # Content type relevance
        if content_type in query_context.boost_content_types:
            score += 0.3
        
        return min(score, 1.0)
    
    def calculate_temporal_score(self, timestamp: int, query_context: QueryContext) -> float:
        """Calculate temporal relevance score"""
        if query_context.temporal_preference == 'any':
            return 1.0
        
        current_time = time.time()
        age_days = (current_time - timestamp) / (24 * 3600)
        
        if query_context.temporal_preference == 'recent':
            # Favor content from last 30 days
            if age_days <= 30:
                return 1.0
            elif age_days <= 90:
                return 0.8
            elif age_days <= 365:
                return 0.6
            else:
                return 0.4
        
        elif query_context.temporal_preference == 'historical':
            # Favor older content
            if age_days >= 365:
                return 1.0
            elif age_days >= 90:
                return 0.8
            else:
                return 0.6
        
        return 1.0
    
    def extract_business_entities(self, content: str) -> List[str]:
        """Extract business entities from content"""
        entities = []
        
        # Extract monetary amounts
        money_pattern = r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*([億万千百十]?)円'
        money_matches = re.findall(money_pattern, content)
        for amount, unit in money_matches:
            entities.append(f"{amount}{unit}円")
        
        # Extract percentages
        percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percent_matches = re.findall(percent_pattern, content)
        for percent in percent_matches:
            entities.append(f"{percent}%")
        
        # Extract years
        year_pattern = r'(\d{4})\s*年'
        year_matches = re.findall(year_pattern, content)
        for year in year_matches:
            entities.append(f"{year}年")
        
        return entities
    
    def enhanced_search(self, query: str, top_k: int = 10, rerank_k: int = 20) -> List[SearchResult]:
        """Enhanced search with advanced ranking"""
        print(f"🔍 Enhanced search: '{query}'")
        
        # Step 1: Analyze query context
        query_context = self.analyze_query_intent(query)
        print(f"   Intent: {query_context.intent}, Domain: {query_context.domain}")
        
        # Step 2: Get initial results with higher k for reranking
        initial_results = self.basic_search(query, top_k=rerank_k)
        if not initial_results:
            return []
        
        # Step 3: Enhanced scoring for each result
        enhanced_results = []
        for result in initial_results:
            # Calculate various scoring components
            keyword_score, matched_keywords = self.calculate_keyword_score(
                result['content'], query_context
            )
            
            business_score = self.calculate_business_relevance_score(
                result['content'], result['content_type'], query_context
            )
            
            content_type_score = self.content_type_weights.get(result['content_type'], 1.0)
            
            temporal_score = self.calculate_temporal_score(
                result.get('timestamp', time.time()), query_context
            )
            
            business_entities = self.extract_business_entities(result['content'])
            
            # Combined scoring with weights
            semantic_weight = 0.4
            keyword_weight = 0.3
            business_weight = 0.2
            content_type_weight = 0.1
            
            combined_score = (
                result['score'] * semantic_weight +
                keyword_score * keyword_weight +
                business_score * business_weight +
                (content_type_score - 1.0) * content_type_weight
            ) * temporal_score
            
            enhanced_result = SearchResult(
                id=result['id'],
                content=result['content'],
                content_type=result['content_type'],
                url=result['url'],
                title=result['title'],
                word_count=result['word_count'],
                timestamp=result.get('timestamp', 0),
                semantic_score=result['score'],
                keyword_score=keyword_score,
                business_relevance_score=business_score,
                content_type_score=content_type_score,
                temporal_score=temporal_score,
                combined_score=combined_score,
                matched_keywords=matched_keywords,
                business_entities=business_entities
            )
            
            enhanced_results.append(enhanced_result)
        
        # Step 4: Sort by combined score and return top_k
        enhanced_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        print(f"   Reranked {len(enhanced_results)} results by combined scoring")
        return enhanced_results[:top_k]
    
    def explain_ranking(self, results: List[SearchResult], limit: int = 3) -> None:
        """Explain ranking decisions for top results"""
        print(f"\n🔍 Ranking Explanation (Top {limit}):")
        print("=" * 60)
        
        for i, result in enumerate(results[:limit], 1):
            print(f"{i}. 【{result.content_type}】Combined Score: {result.combined_score:.4f}")
            print(f"   Title: {result.title[:50]}...")
            print(f"   Scoring Breakdown:")
            print(f"     • Semantic Similarity: {result.semantic_score:.3f}")
            print(f"     • Keyword Relevance: {result.keyword_score:.3f}")
            print(f"     • Business Relevance: {result.business_relevance_score:.3f}")
            print(f"     • Content Type Boost: {result.content_type_score:.3f}")
            print(f"     • Temporal Relevance: {result.temporal_score:.3f}")
            print(f"   Matched Keywords: {', '.join(result.matched_keywords[:5])}")
            print(f"   Business Entities: {', '.join(result.business_entities[:3])}")
            print(f"   Content Preview: {result.content[:100]}...")
            print("-" * 60)

class EnhancedSmartQueryManager:
    """Enhanced smart query manager with optimization features"""
    
    def __init__(self, host='localhost', port='19530'):
        self.host = host
        self.port = port
        self.available_collections = []
        self.query_stats = {
            'total_queries': 0,
            'avg_response_time': 0.0,
            'successful_queries': 0
        }
    
    def connect_and_scan(self) -> bool:
        """Connect and scan available collections"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            print(f"✅ Connected to Milvus: {self.host}:{self.port}")
            
            self.available_collections = utility.list_collections()
            print(f"📋 Found {len(self.available_collections)} collections:")
            
            for i, collection_name in enumerate(self.available_collections, 1):
                collection = Collection(collection_name)
                collection.load()
                print(f"  {i}. {collection_name} ({collection.num_entities} records)")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def auto_select_best_collection(self, query: str) -> Optional[str]:
        """Automatically select the best collection for the query"""
        if not self.available_collections:
            return None
        
        if len(self.available_collections) == 1:
            return self.available_collections[0]
        
        # Simple heuristic: prefer collections with more data
        collection_scores = {}
        for collection_name in self.available_collections:
            try:
                collection = Collection(collection_name)
                score = collection.num_entities
                
                # Boost score for business-related collections
                if 'business' in collection_name.lower() or 'company' in collection_name.lower():
                    score *= 1.2
                
                collection_scores[collection_name] = score
            except:
                collection_scores[collection_name] = 0
        
        best_collection = max(collection_scores, key=collection_scores.get)
        print(f"🎯 Auto-selected collection: {best_collection}")
        return best_collection
    
    def enhanced_query_mode(self, collection_name: str):
        """Enhanced query mode with optimization features"""
        print(f"\n🚀 Enhanced Query Mode - Collection: {collection_name}")
        print("=" * 60)
        
        engine = EnhancedMilvusQueryEngine(collection_name=collection_name)
        if not engine.connect():
            return
        
        try:
            # Display collection statistics
            stats = engine.get_statistics()
            print(f"\n📊 Collection Statistics:")
            print(f"   Records: {stats.get('total_records', 'N/A')}")
            print(f"   Vector Dimension: {stats.get('dimension', 'N/A')}")
            
            if 'content_type_distribution' in stats:
                print("   Content Types:")
                for ct, count in list(stats['content_type_distribution'].items())[:8]:
                    print(f"     {ct}: {count}")
            
            # Interactive enhanced querying
            while True:
                print(f"\n🔍 Enhanced Query Options:")
                print("1. Enhanced semantic search with ranking explanation")
                print("2. Business-focused search")
                print("3. Query performance analysis")
                print("4. View query statistics")
                print("0. Return to main menu")
                
                choice = input("\nSelect option (0-4): ").strip()
                
                if choice == "0":
                    break
                elif choice == "1":
                    query = input("Enter your search query: ").strip()
                    if query:
                        start_time = time.time()
                        results = engine.enhanced_search(query, top_k=5)
                        response_time = time.time() - start_time
                        
                        self._update_query_stats(response_time, len(results) > 0)
                        
                        if results:
                            engine.explain_ranking(results, limit=3)
                            self._print_enhanced_results(results)
                        else:
                            print("No results found")
                
                elif choice == "2":
                    self._business_focused_search(engine)
                
                elif choice == "3":
                    self._query_performance_analysis(engine)
                
                elif choice == "4":
                    self._show_query_statistics()
                
                else:
                    print("❌ Invalid selection")
        
        finally:
            engine.disconnect()
    
    def _business_focused_search(self, engine: EnhancedMilvusQueryEngine):
        """Business-focused search with predefined patterns"""
        print(f"\n💼 Business-Focused Search Patterns:")
        print("1. Company financial information")
        print("2. Business services and solutions")
        print("3. Corporate governance")
        print("4. Recent business developments")
        print("5. Custom business query")
        
        choice = input("Select pattern (1-5): ").strip()
        
        pattern_queries = {
            "1": "会社の売上 利益 財務情報",
            "2": "事業内容 サービス ソリューション",
            "3": "コーポレートガバナンス 企業統治",
            "4": "最新 ニュース 発表",
            "5": None
        }
        
        if choice in pattern_queries:
            if pattern_queries[choice]:
                query = pattern_queries[choice]
                print(f"Searching for: {query}")
            else:
                query = input("Enter your business query: ").strip()
            
            if query:
                start_time = time.time()
                results = engine.enhanced_search(query, top_k=5)
                response_time = time.time() - start_time
                
                self._update_query_stats(response_time, len(results) > 0)
                
                if results:
                    engine.explain_ranking(results, limit=2)
                    self._print_enhanced_results(results)
                else:
                    print("No results found")
    
    def _query_performance_analysis(self, engine: EnhancedMilvusQueryEngine):
        """Analyze query performance with different parameters"""
        test_query = input("Enter test query: ").strip()
        if not test_query:
            return
        
        print(f"\n⚡ Performance Analysis for: '{test_query}'")
        print("-" * 50)
        
        # Test different top_k values
        for top_k in [5, 10, 20]:
            start_time = time.time()
            results = engine.enhanced_search(test_query, top_k=top_k)
            response_time = time.time() - start_time
            
            print(f"top_k={top_k}: {response_time:.3f}s, {len(results)} results")
        
        # Test basic vs enhanced search
        start_time = time.time()
        basic_results = engine.basic_search(test_query, top_k=5)
        basic_time = time.time() - start_time
        
        start_time = time.time()
        enhanced_results = engine.enhanced_search(test_query, top_k=5)
        enhanced_time = time.time() - start_time
        
        print(f"\nComparison:")
        print(f"Basic search: {basic_time:.3f}s")
        print(f"Enhanced search: {enhanced_time:.3f}s")
        print(f"Enhancement overhead: {(enhanced_time - basic_time):.3f}s")
    
    def _update_query_stats(self, response_time: float, successful: bool):
        """Update query statistics"""
        self.query_stats['total_queries'] += 1
        
        # Update average response time
        current_avg = self.query_stats['avg_response_time']
        total = self.query_stats['total_queries']
        self.query_stats['avg_response_time'] = (current_avg * (total - 1) + response_time) / total
        
        if successful:
            self.query_stats['successful_queries'] += 1
    
    def _show_query_statistics(self):
        """Show query performance statistics"""
        stats = self.query_stats
        print(f"\n📈 Query Performance Statistics:")
        print(f"   Total Queries: {stats['total_queries']}")
        print(f"   Successful Queries: {stats['successful_queries']}")
        print(f"   Success Rate: {stats['successful_queries']/max(stats['total_queries'], 1)*100:.1f}%")
        print(f"   Average Response Time: {stats['avg_response_time']:.3f}s")
    
    def _print_enhanced_results(self, results: List[SearchResult]):
        """Print enhanced search results"""
        print(f"\n📋 Enhanced Search Results ({len(results)} results):")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. 【{result.content_type}】Score: {result.combined_score:.4f}")
            print(f"   Title: {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Business Entities: {', '.join(result.business_entities[:3])}")
            content = result.content
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"   Content: {content}")
            print("-" * 80)
    
    def main_menu(self):
        """Enhanced main menu"""
        if not self.connect_and_scan():
            return
        
        while True:
            print(f"\n🧠 Enhanced Smart Query System")
            print("=" * 60)
            print("Select query mode:")
            print("1. 🔍 Enhanced Python vector search")
            print("2. 🤖 LLaMA intelligent Q&A")
            print("3. 📊 Collection analysis")
            print("4. ⚙️  System optimization")
            print("0. Exit")
            
            choice = input("\nSelect option (0-4): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                # Auto-select best collection or let user choose
                collection_name = self.auto_select_best_collection("")
                if collection_name:
                    self.enhanced_query_mode(collection_name)
            elif choice == "2":
                # Keep existing LLaMA functionality
                collection_name = self.auto_select_best_collection("")
                if collection_name:
                    self._llama_query_mode(collection_name)
            elif choice == "3":
                self._collection_analysis()
            elif choice == "4":
                self._system_optimization()
            else:
                print("❌ Invalid selection")
    
    def _llama_query_mode(self, collection_name: str):
        """LLaMA query mode (simplified)"""
        print(f"\n🤖 LLaMA Intelligent Q&A - Collection: {collection_name}")
        try:
            engine = LLaMAQueryEngine(
                model_type='ollama', 
                model_name='llama3.2:3b',
                collection_name=collection_name
            )
            
            if engine.connect_to_milvus():
                engine.interactive_chat()
        except Exception as e:
            print(f"❌ LLaMA mode initialization failed: {e}")
    
    def _collection_analysis(self):
        """Analyze collection quality and optimization opportunities"""
        print(f"\n📊 Collection Analysis")
        print("-" * 40)
        
        for collection_name in self.available_collections:
            try:
                engine = EnhancedMilvusQueryEngine(collection_name=collection_name)
                if engine.connect():
                    stats = engine.get_statistics()
                    
                    print(f"\nCollection: {collection_name}")
                    print(f"  Records: {stats.get('total_records', 0)}")
                    print(f"  Content Types: {len(stats.get('content_type_distribution', {}))}")
                    
                    # Quality metrics
                    if 'content_type_distribution' in stats:
                        total_records = stats.get('total_records', 1)
                        distribution = stats['content_type_distribution']
                        
                        # Calculate diversity
                        entropy = 0
                        for count in distribution.values():
                            p = count / total_records
                            if p > 0:
                                entropy -= p * math.log2(p)
                        
                        print(f"  Content Diversity: {entropy:.2f} bits")
                        
                        # Most common content types
                        sorted_types = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
                        print(f"  Top Content Types: {', '.join([f'{ct}({count})' for ct, count in sorted_types[:3]])}")
                    
                    engine.disconnect()
            except Exception as e:
                print(f"  Error analyzing {collection_name}: {e}")
    
    def _system_optimization(self):
        """System optimization recommendations"""
        print(f"\n⚙️  System Optimization Recommendations")
        print("=" * 50)
        
        recommendations = [
            "1. Data Quality Improvements:",
            "   • Implement semantic chunking instead of fixed-size chunks",
            "   • Add deduplication for similar content blocks",
            "   • Enhance business entity extraction",
            "",
            "2. Vector Embedding Optimization:",
            "   • Consider domain-specific embedding models",
            "   • Implement hybrid embeddings (semantic + keyword)",
            "   • Add embedding quality validation",
            "",
            "3. Milvus Configuration:",
            "   • Optimize HNSW parameters for your data size",
            "   • Consider multiple indexes for different content types",
            "   • Implement query result caching",
            "",
            "4. Query Strategy Enhancement:",
            "   • Add query expansion and refinement",
            "   • Implement user feedback learning",
            "   • Add query intent classification",
            "",
            "5. Performance Monitoring:",
            "   • Set up query performance tracking",
            "   • Monitor vector index health",
            "   • Implement A/B testing for ranking algorithms"
        ]
        
        for recommendation in recommendations:
            print(recommendation)

def main():
    """Main function with enhanced capabilities"""
    print("🚀 Enhanced Smart Query System")
    print("=" * 70)
    print("Advanced AI training optimization with:")
    print("• Intelligent query intent analysis")
    print("• Multi-component ranking system")
    print("• Business context awareness")
    print("• Performance monitoring")
    print("• Optimization recommendations")
    print("-" * 70)
    
    manager = EnhancedSmartQueryManager()
    manager.main_menu()

if __name__ == "__main__":
    main()