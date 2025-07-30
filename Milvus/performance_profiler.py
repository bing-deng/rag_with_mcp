#!/usr/bin/env python3
"""
性能分析器 - 测量各个组件的执行时间
"""

import time
import functools
from typing import Dict, Any

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
    
    def time_function(self, name: str):
        """装饰器：测量函数执行时间"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.time() - start_time
                    self.record_timing(name, elapsed)
                    print(f"⏱️  {name}: {elapsed:.3f}s")
            return wrapper
        return decorator
    
    def record_timing(self, name: str, elapsed: float):
        """记录时间"""
        if name not in self.timings:
            self.timings[name] = []
            self.call_counts[name] = 0
        
        self.timings[name].append(elapsed)
        self.call_counts[name] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {}
        for name, times in self.timings.items():
            stats[name] = {
                'avg_time': sum(times) / len(times),
                'total_time': sum(times),
                'call_count': self.call_counts[name],
                'min_time': min(times),
                'max_time': max(times)
            }
        return stats
    
    def print_report(self):
        """打印性能报告"""
        print("\n📊 性能分析报告")
        print("=" * 60)
        stats = self.get_stats()
        
        # 按平均时间排序
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['avg_time'], reverse=True)
        
        for name, data in sorted_stats:
            print(f"{name:25} | 平均: {data['avg_time']:.3f}s | 总计: {data['total_time']:.3f}s | 调用: {data['call_count']}次")

# 全局性能分析器
profiler = PerformanceProfiler()

def get_profiler():
    """获取全局性能分析器"""
    return profiler