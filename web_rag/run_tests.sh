#!/bin/bash

echo "=== 导入测试脚本 ==="
echo ""

cd "$(dirname "$0")"

echo "当前目录: $(pwd)"
echo "Python版本: $(python --version)"
echo ""

python test_imports.py 