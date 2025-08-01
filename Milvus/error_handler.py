#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的错误处理和日志系统
提供统一的异常处理、日志记录和恢复机制
"""

import os
import sys
import time
import logging
import traceback
import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import json

class ErrorLevel(Enum):
    """错误级别"""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"

class ErrorCategory(Enum):
    """错误类别"""
    PDF_PROCESSING = "PDF_PROCESSING"
    VECTORIZATION = "VECTORIZATION"
    DATABASE = "DATABASE"
    API = "API"
    SYSTEM = "SYSTEM"
    NETWORK = "NETWORK"
    VALIDATION = "VALIDATION"

@dataclass
class ErrorRecord:
    """错误记录"""
    timestamp: str
    level: str
    category: str
    message: str
    exception_type: str
    traceback_info: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_success: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class EnhancedLogger:
    """增强的日志记录器"""
    
    def __init__(self, name: str = "PDF_Search_System", log_file: str = "system.log"):
        self.name = name
        self.log_file = log_file
        self.error_records: List[ErrorRecord] = []
        self._lock = threading.Lock()
        
        # 配置日志
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"增强日志系统初始化完成: {name}")
    
    def log_error(self, 
                  level: ErrorLevel, 
                  category: ErrorCategory, 
                  message: str,
                  exception: Optional[Exception] = None,
                  context: Optional[Dict[str, Any]] = None,
                  recovery_attempted: bool = False,
                  recovery_success: bool = False):
        """记录错误"""
        
        timestamp = datetime.now().isoformat()
        exception_type = type(exception).__name__ if exception else "None"
        traceback_info = traceback.format_exc() if exception else ""
        context = context or {}
        
        # 创建错误记录
        error_record = ErrorRecord(
            timestamp=timestamp,
            level=level.value,
            category=category.value,
            message=message,
            exception_type=exception_type,
            traceback_info=traceback_info,
            context=context,
            recovery_attempted=recovery_attempted,
            recovery_success=recovery_success
        )
        
        # 线程安全地添加记录
        with self._lock:
            self.error_records.append(error_record)
        
        # 记录日志
        log_message = f"[{category.value}] {message}"
        if exception:
            log_message += f" - Exception: {exception}"
        
        if level == ErrorLevel.CRITICAL:
            self.logger.critical(log_message)
        elif level == ErrorLevel.ERROR:
            self.logger.error(log_message)
        elif level == ErrorLevel.WARNING:
            self.logger.warning(log_message)
        elif level == ErrorLevel.INFO:
            self.logger.info(log_message)
        elif level == ErrorLevel.DEBUG:
            self.logger.debug(log_message)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        with self._lock:
            total_errors = len(self.error_records)
            error_by_level = {}
            error_by_category = {}
            recent_errors = []
            
            for record in self.error_records:
                # 按级别统计
                level = record.level
                error_by_level[level] = error_by_level.get(level, 0) + 1
                
                # 按类别统计
                category = record.category
                error_by_category[category] = error_by_category.get(category, 0) + 1
                
                # 最近的错误（最近10个）
                if len(recent_errors) < 10:
                    recent_errors.append({
                        'timestamp': record.timestamp,
                        'level': record.level,
                        'category': record.category,
                        'message': record.message[:100] + ('...' if len(record.message) > 100 else '')
                    })
            
            return {
                'total_errors': total_errors,
                'error_by_level': error_by_level,
                'error_by_category': error_by_category,
                'recent_errors': recent_errors,
                'log_file': self.log_file
            }
    
    def export_error_report(self, output_file: str = None) -> str:
        """导出错误报告"""
        if not output_file:
            output_file = f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with self._lock:
            report_data = {
                'generated_at': datetime.now().isoformat(),
                'system_name': self.name,
                'summary': self.get_error_summary(),
                'detailed_errors': [record.to_dict() for record in self.error_records]
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"错误报告已导出: {output_file}")
        return output_file

# 全局日志实例
enhanced_logger = EnhancedLogger()

def error_handler(category: ErrorCategory, 
                 level: ErrorLevel = ErrorLevel.ERROR,
                 reraise: bool = True,
                 default_return: Any = None,
                 recovery_func: Optional[Callable] = None):
    """错误处理装饰器"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 记录错误
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],
                    'kwargs': str(kwargs)[:200]
                }
                
                recovery_attempted = False
                recovery_success = False
                
                # 尝试恢复
                if recovery_func:
                    try:
                        recovery_attempted = True
                        enhanced_logger.log_error(
                            level=ErrorLevel.INFO,
                            category=category,
                            message=f"尝试恢复函数 {func.__name__} 的错误",
                            context=context
                        )
                        
                        result = recovery_func(*args, **kwargs)
                        recovery_success = True
                        
                        enhanced_logger.log_error(
                            level=ErrorLevel.INFO,
                            category=category,
                            message=f"函数 {func.__name__} 恢复成功",
                            context=context,
                            recovery_attempted=True,
                            recovery_success=True
                        )
                        
                        return result
                        
                    except Exception as recovery_error:
                        enhanced_logger.log_error(
                            level=ErrorLevel.ERROR,
                            category=category,
                            message=f"函数 {func.__name__} 恢复失败",
                            exception=recovery_error,
                            context=context,
                            recovery_attempted=True,
                            recovery_success=False
                        )
                
                # 记录原始错误
                enhanced_logger.log_error(
                    level=level,
                    category=category,
                    message=f"函数 {func.__name__} 执行失败: {str(e)}",
                    exception=e,
                    context=context,
                    recovery_attempted=recovery_attempted,
                    recovery_success=recovery_success
                )
                
                if reraise:
                    raise
                else:
                    return default_return
        
        return wrapper
    return decorator

def validate_input(validation_func: Callable, error_message: str = "输入验证失败"):
    """输入验证装饰器"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if not validation_func(*args, **kwargs):
                    raise ValueError(error_message)
                return func(*args, **kwargs)
            except Exception as e:
                enhanced_logger.log_error(
                    level=ErrorLevel.WARNING,
                    category=ErrorCategory.VALIDATION,
                    message=f"输入验证失败: {error_message}",
                    exception=e,
                    context={'function': func.__name__}
                )
                raise
        
        return wrapper
    return decorator

def retry_on_failure(max_retries: int = 3, 
                    delay: float = 1.0, 
                    category: ErrorCategory = ErrorCategory.SYSTEM):
    """失败重试装饰器"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        enhanced_logger.log_error(
                            level=ErrorLevel.WARNING,
                            category=category,
                            message=f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败，将重试",
                            exception=e,
                            context={'attempt': attempt + 1, 'max_retries': max_retries}
                        )
                        time.sleep(delay * (2 ** attempt))  # 指数退避
                    else:
                        enhanced_logger.log_error(
                            level=ErrorLevel.ERROR,
                            category=category,
                            message=f"函数 {func.__name__} 重试 {max_retries} 次后仍然失败",
                            exception=e,
                            context={'total_attempts': max_retries + 1}
                        )
            
            raise last_exception
        
        return wrapper
    return decorator

class SafeExecutor:
    """安全执行器"""
    
    @staticmethod
    def execute_safely(func: Callable, 
                      *args, 
                      category: ErrorCategory = ErrorCategory.SYSTEM,
                      default_return: Any = None,
                      **kwargs) -> Any:
        """安全执行函数"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            enhanced_logger.log_error(
                level=ErrorLevel.ERROR,
                category=category,
                message=f"安全执行失败: {func.__name__}",
                exception=e,
                context={'args': str(args)[:100], 'kwargs': str(kwargs)[:100]}
            )
            return default_return
    
    @staticmethod
    def execute_with_timeout(func: Callable, 
                           timeout: float,
                           *args,
                           category: ErrorCategory = ErrorCategory.SYSTEM,
                           **kwargs) -> Any:
        """带超时的执行"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"函数 {func.__name__} 执行超时 ({timeout}秒)")
        
        # 设置超时信号
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # 取消超时
            return result
        except TimeoutError as e:
            enhanced_logger.log_error(
                level=ErrorLevel.ERROR,
                category=category,
                message=f"函数执行超时: {func.__name__}",
                exception=e,
                context={'timeout': timeout}
            )
            raise
        except Exception as e:
            signal.alarm(0)  # 取消超时
            enhanced_logger.log_error(
                level=ErrorLevel.ERROR,
                category=category,
                message=f"函数执行异常: {func.__name__}",
                exception=e
            )
            raise
        finally:
            signal.signal(signal.SIGALRM, old_handler)

# 常用的恢复函数
def pdf_processing_recovery(*args, **kwargs):
    """PDF处理恢复函数"""
    enhanced_logger.log_error(
        level=ErrorLevel.INFO,
        category=ErrorCategory.PDF_PROCESSING,
        message="尝试使用备用PDF处理方法",
        context={'recovery_method': 'fallback_processing'}
    )
    # 这里可以实现备用的PDF处理逻辑
    return []

def database_connection_recovery(*args, **kwargs):
    """数据库连接恢复函数"""
    enhanced_logger.log_error(
        level=ErrorLevel.INFO,
        category=ErrorCategory.DATABASE,
        message="尝试重新连接数据库",
        context={'recovery_method': 'reconnect'}
    )
    # 这里可以实现数据库重连逻辑
    time.sleep(2)  # 等待2秒后重试
    return None

# 预定义的错误处理装饰器
pdf_error_handler = error_handler(
    category=ErrorCategory.PDF_PROCESSING,
    level=ErrorLevel.ERROR,
    recovery_func=pdf_processing_recovery
)

db_error_handler = error_handler(
    category=ErrorCategory.DATABASE,
    level=ErrorLevel.ERROR,
    recovery_func=database_connection_recovery
)

api_error_handler = error_handler(
    category=ErrorCategory.API,
    level=ErrorLevel.WARNING,
    reraise=False,
    default_return={"error": "API调用失败"}
)

if __name__ == "__main__":
    # 测试错误处理系统
    @pdf_error_handler
    def test_pdf_function():
        raise ValueError("测试PDF处理错误")
    
    @retry_on_failure(max_retries=2)
    def test_retry_function():
        import random
        if random.random() < 0.7:  # 70%的概率失败
            raise ConnectionError("随机连接错误")
        return "成功"
    
    # 测试
    try:
        test_pdf_function()
    except:
        pass
    
    try:
        result = test_retry_function()
        print(f"重试测试结果: {result}")
    except:
        pass
    
    # 输出错误摘要
    summary = enhanced_logger.get_error_summary()
    print("\n错误摘要:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    
    # 导出错误报告
    report_file = enhanced_logger.export_error_report()
    print(f"\n错误报告已生成: {report_file}")