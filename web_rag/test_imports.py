#!/usr/bin/env python3
"""测试所有导入是否正常工作"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"项目根目录: {project_root}")
print(f"Python路径: {sys.path[:3]}...")

# 测试导入
tests = [
    ("Flask", "flask", "Flask"),
    ("boto3", "boto3", None),
    ("PyPDF2", "PyPDF2", None),
    ("pdfplumber", "pdfplumber", None),
    ("weaviate", "weaviate", None),
    ("numpy", "numpy", None),
]

print("\n=== 测试基础依赖导入 ===")
for name, module, attr in tests:
    try:
        imported = __import__(module)
        if attr:
            getattr(imported, attr)
        print(f"✅ {name}: OK")
    except ImportError as e:
        print(f"❌ {name}: 失败 - {e}")
    except AttributeError as e:
        print(f"❌ {name}: 属性错误 - {e}")

print("\n=== 测试项目模块导入 ===")

# 测试bedrock模块
try:
    from bedrock.bedrock_usage import TokyoBedrockService
    print("✅ bedrock.bedrock_usage: OK")
except ImportError as e:
    print(f"❌ bedrock.bedrock_usage: 失败 - {e}")

# 测试weaviate客户端
try:
    # 方法1: 直接从weaviate目录导入
    weaviate_path = os.path.join(project_root, 'weaviate')
    if weaviate_path not in sys.path:
        sys.path.insert(0, weaviate_path)
    
    from weaviate_client import WeaviateRAGClient
    print("✅ weaviate_client: OK")
except ImportError as e:
    print(f"❌ weaviate_client: 失败 - {e}")
    print(f"   Weaviate目录: {weaviate_path}")
    print(f"   目录存在: {os.path.exists(weaviate_path)}")
    
    # 列出weaviate目录内容
    if os.path.exists(weaviate_path):
        print(f"   目录内容: {os.listdir(weaviate_path)}")

# 测试PDF处理器
try:
    from pdf_processor import PDFProcessor
    print("✅ pdf_processor: OK")
except ImportError as e:
    print(f"❌ pdf_processor: 失败 - {e}")

print("\n=== 测试完成 ===") 