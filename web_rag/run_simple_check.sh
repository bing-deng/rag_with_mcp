#!/bin/bash

echo "🔧 シンプルヘルスチェック実行中..."

# Pythonバイトコードキャッシュをクリア
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 仮想環境アクティベート（存在する場合）
if [ -f "../bedrock/venv/bin/activate.fish" ]; then
    echo "✅ 仮想環境をアクティベート"
    source ../bedrock/venv/bin/activate.fish
fi

# シンプルチェック実行
python3 simple_health_check.py

echo "🏁 チェック完了" 