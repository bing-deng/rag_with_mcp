<img width="1182" height="872" alt="image" src="https://github.com/user-attachments/assets/c704b4d8-56e9-4288-994c-748b74fb5fe6" />



```
<svg viewBox="0 0 1400 1000" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .title { font-family: "Noto Sans JP", Arial, sans-serif; font-size: 24px; font-weight: bold; fill: #2c3e50; }
      .subtitle { font-family: "Noto Sans JP", Arial, sans-serif; font-size: 18px; font-weight: bold; fill: #34495e; }
      .section-title { font-family: "Noto Sans JP", Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #e74c3c; }
      .text { font-family: "Noto Sans JP", Arial, sans-serif; font-size: 14px; fill: #2c3e50; }
      .small-text { font-family: "Noto Sans JP", Arial, sans-serif; font-size: 12px; fill: #7f8c8d; }
      .step-text { font-family: "Noto Sans JP", Arial, sans-serif; font-size: 11px; fill: #2c3e50; }
      .time-text { font-family: "Noto Sans JP", Arial, sans-serif; font-size: 13px; font-weight: bold; fill: #e74c3c; }
      .rag-color { fill: #3498db; stroke: #2980b9; }
      .mcp-color { fill: #e74c3c; stroke: #c0392b; }
      .process-color { fill: #f39c12; stroke: #e67e22; }
      .success { fill: #27ae60; stroke: #2ecc71; }
      .warning { fill: #f39c12; stroke: #e67e22; }
      .error { fill: #e74c3c; stroke: #c0392b; }
      .arrow { stroke: #34495e; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }
      .heavy-arrow { stroke: #e74c3c; stroke-width: 3; fill: none; marker-end: url(#heavy-arrowhead); }
      .dashed { stroke-dasharray: 8,4; }
      .dotted { stroke-dasharray: 3,3; }
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#34495e"/>
    </marker>
    <marker id="heavy-arrowhead" markerWidth="12" markerHeight="9" refX="11" refY="4.5" orient="auto">
      <polygon points="0 0, 12 4.5, 0 9" fill="#e74c3c"/>
    </marker>
  </defs>
  
  <!-- Main Title -->
  <text x="700" y="30" text-anchor="middle" class="title">RAG vs MCP データ更新メカニズムの比較</text>
  
  <!-- RAG データ更新プロセス -->
  <g>
    <rect x="50" y="60" width="650" height="450" rx="15" class="rag-color" fill-opacity="0.1" stroke-width="3"/>
    <text x="375" y="85" text-anchor="middle" class="subtitle">RAG データ更新プロセス（重い・複雑）</text>
    
    <!-- Step 1: 新データ収集 -->
    <g>
      <rect x="80" y="110" width="120" height="50" rx="8" class="process-color"/>
      <text x="140" y="130" text-anchor="middle" class="text" fill="white">1. 新データ収集</text>
      <text x="140" y="145" text-anchor="middle" class="step-text" fill="white">（しんデータしゅうしゅう）</text>
      
      <text x="80" y="175" class="small-text">• 新しい文書の追加</text>
      <text x="80" y="190" class="small-text">• 既存文書の修正</text>
      <text x="80" y="205" class="small-text">• 削除された文書の特定</text>
    </g>
    
    <!-- Arrow 1 -->
    <path d="M 200 135 L 240 135" class="arrow"/>
    
    <!-- Step 2: 前処理 -->
    <g>
      <rect x="240" y="110" width="120" height="50" rx="8" class="process-color"/>
      <text x="300" y="130" text-anchor="middle" class="text" fill="white">2. 前処理</text>
      <text x="300" y="145" text-anchor="middle" class="step-text" fill="white">（まえしょり）</text>
      
      <text x="240" y="175" class="small-text">• テキスト抽出</text>
      <text x="240" y="190" class="small-text">• クリーニング処理</text>
      <text x="240" y="205" class="small-text">• チャンク分割</text>
    </g>
    
    <!-- Arrow 2 -->
    <path d="M 360 135 L 400 135" class="arrow"/>
    
    <!-- Step 3: ベクトル化 -->
    <g>
      <rect x="400" y="110" width="120" height="50" rx="8" class="process-color"/>
      <text x="460" y="130" text-anchor="middle" class="text" fill="white">3. ベクトル化</text>
      <text x="460" y="145" text-anchor="middle" class="step-text" fill="white">（ベクトルか）</text>
      
      <text x="400" y="175" class="small-text">• 埋め込みモデル実行</text>
      <text x="400" y="190" class="small-text">• ベクトル生成</text>
      <text x="400" y="205" class="small-text">• 次元削減処理</text>
    </g>
    
    <!-- Arrow 3 -->
    <path d="M 520 135 L 560 135" class="arrow"/>
    
    <!-- Step 4: インデックス更新 -->
    <g>
      <rect x="560" y="110" width="120" height="50" rx="8" class="warning"/>
      <text x="620" y="130" text-anchor="middle" class="text" fill="white">4. インデックス更新</text>
      <text x="620" y="145" text-anchor="middle" class="step-text" fill="white">（インデックスこうしん）</text>
      
      <text x="560" y="175" class="small-text">• 既存インデックス削除</text>
      <text x="560" y="190" class="small-text">• 新インデックス構築</text>
      <text x="560" y="205" class="small-text">• 検索最適化</text>
    </g>
    
    <!-- 処理時間表示 -->
    <rect x="80" y="230" width="600" height="40" rx="5" class="warning" fill-opacity="0.2"/>
    <text x="380" y="250" text-anchor="middle" class="time-text">⏰ 処理時間：数時間〜数日</text>
    <text x="380" y="265" text-anchor="middle" class="small-text">大量データの場合、システム全体の停止が必要な場合もある</text>
    
    <!-- RAGの課題 -->
    <text x="80" y="295" class="section-title">RAG更新の課題と特徴：</text>
    
    <!-- 課題1: バッチ処理 -->
    <g>
      <rect x="80" y="310" width="180" height="80" rx="5" class="error" fill-opacity="0.1"/>
      <text x="170" y="330" text-anchor="middle" class="text">バッチ処理必須</text>
      <text x="90" y="350" class="step-text">• リアルタイム更新困難</text>
      <text x="90" y="365" class="step-text">• 夜間バッチで処理</text>
      <text x="90" y="380" class="step-text">• 更新中は検索不可</text>
    </g>
    
    <!-- 課題2: 高コスト -->
    <g>
      <rect x="280" y="310" width="180" height="80" rx="5" class="error" fill-opacity="0.1"/>
      <text x="370" y="330" text-anchor="middle" class="text">高い計算コスト</text>
      <text x="290" y="350" class="step-text">• GPU/CPU集約的処理</text>
      <text x="290" y="365" class="step-text">• 電力消費大</text>
      <text x="290" y="380" class="step-text">• インフラコスト高</text>
    </g>
    
    <!-- 課題3: データ整合性 -->
    <g>
      <rect x="480" y="310" width="180" height="80" rx="5" class="error" fill-opacity="0.1"/>
      <text x="570" y="330" text-anchor="middle" class="text">データ整合性管理</text>
      <text x="490" y="350" class="step-text">• バージョン管理複雑</text>
      <text x="490" y="365" class="step-text">• 部分更新困難</text>
      <text x="490" y="380" class="step-text">• ロールバック複雑</text>
    </g>
    
    <!-- 更新頻度 -->
    <rect x="80" y="410" width="580" height="50" rx="5" class="rag-color" fill-opacity="0.2"/>
    <text x="370" y="430" text-anchor="middle" class="text">典型的な更新頻度：週1回〜月1回</text>
    <text x="370" y="445" text-anchor="middle" class="small-text">日本企業例：トヨタの技術文書（月1回）、銀行の規制文書（週1回）</text>
  </g>
  
  <!-- MCP データ更新プロセス -->
  <g>
    <rect x="750" y="60" width="600" height="450" rx="15" class="mcp-color" fill-opacity="0.1" stroke-width="3"/>
    <text x="1050" y="85" text-anchor="middle" class="subtitle">MCP データ更新プロセス（軽い・シンプル）</text>
    
    <!-- リアルタイムアクセス -->
    <g>
      <rect x="780" y="110" width="200" height="60" rx="8" class="success"/>
      <text x="880" y="135" text-anchor="middle" class="text" fill="white">リアルタイムアクセス</text>
      <text x="880" y="150" text-anchor="middle" class="step-text" fill="white">（リアルタイムアクセス）</text>
      
      <text x="780" y="185" class="small-text">• データソースから直接取得</text>
      <text x="780" y="200" class="small-text">• 常に最新データを参照</text>
    </g>
    
    <!-- API呼び出し例 -->
    <g>
      <rect x="1000" y="110" width="150" height="60" rx="8" class="mcp-color"/>
      <text x="1075" y="135" text-anchor="middle" class="text" fill="white">API呼び出し</text>
      <text x="1075" y="150" text-anchor="middle" class="step-text" fill="white">（APIよびだし）</text>
      
      <text x="1000" y="185" class="small-text">• RESTful API</text>
      <text x="1000" y="200" class="small-text">• GraphQL</text>
    </g>
    
    <!-- 処理時間表示 -->
    <rect x="780" y="220" width="370" height="40" rx="5" class="success" fill-opacity="0.2"/>
    <text x="965" y="240" text-anchor="middle" class="time-text">⚡ 処理時間：数ミリ秒〜数秒</text>
    <text x="965" y="255" text-anchor="middle" class="small-text">ネットワーク遅延のみが制限要因</text>
    
    <!-- MCPの利点 -->
    <text x="780" y="285" class="section-title">MCP更新の利点と特徴：</text>
    
    <!-- 利点1: 即座更新 -->
    <g>
      <rect x="780" y="300" width="170" height="80" rx="5" class="success" fill-opacity="0.1"/>
      <text x="865" y="320" text-anchor="middle" class="text">即座の更新反映</text>
      <text x="790" y="340" class="step-text">• データ変更即座反映</text>
      <text x="790" y="355" class="step-text">• 前処理不要</text>
      <text x="790" y="370" class="step-text">• 24時間365日対応</text>
    </g>
    
    <!-- 利点2: 低コスト -->
    <g>
      <rect x="970" y="300" width="170" height="80" rx="5" class="success" fill-opacity="0.1"/>
      <text x="1055" y="320" text-anchor="middle" class="text">低い運用コスト</text>
      <text x="980" y="340" class="step-text">• 計算処理最小限</text>
      <text x="980" y="355" class="step-text">• ストレージ不要</text>
      <text x="980" y="370" class="step-text">• メンテナンス簡単</text>
    </g>
    
    <!-- データフロー図 -->
    <text x="780" y="410" class="section-title">MCPデータフロー：</text>
    
    <!-- User Request -->
    <ellipse cx="820" cy="435" rx="35" ry="15" fill="#ecf0f1" stroke="#bdc3c7"/>
    <text x="820" y="440" text-anchor="middle" class="step-text">要求</text>
    
    <!-- Arrow -->
    <path d="M 855 435 L 885 435" class="heavy-arrow"/>
    
    <!-- MCP Protocol -->
    <rect x="885" y="420" width="80" height="30" rx="5" class="mcp-color"/>
    <text x="925" y="440" text-anchor="middle" class="step-text" fill="white">MCP</text>
    
    <!-- Arrow -->
    <path d="M 965 435 L 995 435" class="heavy-arrow"/>
    
    <!-- Live Data -->
    <rect x="995" y="420" width="80" height="30" rx="5" class="success"/>
    <text x="1035" y="440" text-anchor="middle" class="step-text" fill="white">最新データ</text>
    
    <!-- Return Arrow -->
    <path d="M 995 450 L 855 450" class="heavy-arrow"/>
    <text x="925" y="470" text-anchor="middle" class="small-text">瞬時レスポンス</text>
  </g>
  
  <!-- 比較表 -->
  <g>
    <rect x="50" y="540" width="1300" height="220" rx="10" fill="#f8f9fa" stroke="#dee2e6" stroke-width="2"/>
    <text x="700" y="565" text-anchor="middle" class="subtitle">更新メカニズム詳細比較表</text>
    
    <!-- ヘッダー -->
    <rect x="80" y="580" width="150" height="30" rx="3" fill="#6c757d"/>
    <text x="155" y="600" text-anchor="middle" class="text" fill="white">比較項目</text>
    
    <rect x="230" y="580" width="400" height="30" rx="3" class="rag-color"/>
    <text x="430" y="600" text-anchor="middle" class="text" fill="white">RAG（検索拡張生成）</text>
    
    <rect x="630" y="580" width="400" height="30" rx="3" class="mcp-color"/>
    <text x="830" y="600" text-anchor="middle" class="text" fill="white">MCP（モデルコンテキストプロトコル）</text>
    
    <rect x="1030" y="580" width="150" height="30" rx="3" fill="#28a745"/>
    <text x="1105" y="600" text-anchor="middle" class="text" fill="white">優位性</text>
    
    <!-- 行1: 更新頻度 -->
    <rect x="80" y="610" width="150" height="25" fill="#e9ecef"/>
    <text x="155" y="627" text-anchor="middle" class="small-text">更新頻度</text>
    
    <rect x="230" y="610" width="400" height="25" fill="#e3f2fd"/>
    <text x="430" y="627" text-anchor="middle" class="small-text">週1回〜月1回（バッチ処理）</text>
    
    <rect x="630" y="610" width="400" height="25" fill="#ffebee"/>
    <text x="830" y="627" text-anchor="middle" class="small-text">リアルタイム（常時最新）</text>
    
    <rect x="1030" y="610" width="150" height="25" fill="#f1f8e9"/>
    <text x="1105" y="627" text-anchor="middle" class="small-text">MCP</text>
    
    <!-- 行2: 処理時間 -->
    <rect x="80" y="635" width="150" height="25" fill="#e9ecef"/>
    <text x="155" y="652" text-anchor="middle" class="small-text">処理時間</text>
    
    <rect x="230" y="635" width="400" height="25" fill="#e3f2fd"/>
    <text x="430" y="652" text-anchor="middle" class="small-text">数時間〜数日</text>
    
    <rect x="630" y="635" width="400" height="25" fill="#ffebee"/>
    <text x="830" y="652" text-anchor="middle" class="small-text">数ミリ秒〜数秒</text>
    
    <rect x="1030" y="635" width="150" height="25" fill="#f1f8e9"/>
    <text x="1105" y="652" text-anchor="middle" class="small-text">MCP</text>
    
    <!-- 行3: コスト -->
    <rect x="80" y="660" width="150" height="25" fill="#e9ecef"/>
    <text x="155" y="677" text-anchor="middle" class="small-text">運用コスト</text>
    
    <rect x="230" y="660" width="400" height="25" fill="#e3f2fd"/>
    <text x="430" y="677" text-anchor="middle" class="small-text">高い（GPU/CPU集約的）</text>
    
    <rect x="630" y="660" width="400" height="25" fill="#ffebee"/>
    <text x="830" y="677" text-anchor="middle" class="small-text">低い（ネットワーク呼び出しのみ）</text>
    
    <rect x="1030" y="660" width="150" height="25" fill="#f1f8e9"/>
    <text x="1105" y="677" text-anchor="middle" class="small-text">MCP</text>
    
    <!-- 行4: 複雑性 -->
    <rect x="80" y="685" width="150" height="25" fill="#e9ecef"/>
    <text x="155" y="702" text-anchor="middle" class="small-text">実装複雑性</text>
    
    <rect x="230" y="685" width="400" height="25" fill="#e3f2fd"/>
    <text x="430" y="702" text-anchor="middle" class="small-text">高い（パイプライン管理必要）</text>
    
    <rect x="630" y="685" width="400" height="25" fill="#ffebee"/>
    <text x="830" y="702" text-anchor="middle" class="small-text">低い（API呼び出しのみ）</text>
    
    <rect x="1030" y="685" width="150" height="25" fill="#f1f8e9"/>
    <text x="1105" y="702" text-anchor="middle" class="small-text">MCP</text>
    
    <!-- 行5: データ完全性 -->
    <rect x="80" y="710" width="150" height="25" fill="#e9ecef"/>
    <text x="155" y="727" text-anchor="middle" class="small-text">データ完全性</text>
    
    <rect x="230" y="710" width="400" height="25" fill="#e3f2fd"/>
    <text x="430" y="727" text-anchor="middle" class="small-text">高い（全データ検索可能）</text>
    
    <rect x="630" y="710" width="400" height="25" fill="#ffebee"/>
    <text x="830" y="727" text-anchor="middle" class="small-text">中程度（API制限依存）</text>
    
    <rect x="1030" y="710" width="150" height="25" fill="#f1f8e9"/>
    <text x="1105" y="727" text-anchor="middle" class="small-text">RAG</text>
  </g>
  
  <!-- 日本企業での実装例 -->
  <g>
    <text x="700" y="790" text-anchor="middle" class="subtitle">日本企業での実装パターン</text>
    
    <!-- RAG更新例 -->
    <rect x="80" y="810" width="300" height="120" rx="8" class="rag-color" fill-opacity="0.1"/>
    <text x="230" y="830" text-anchor="middle" class="text">RAG更新の実装例</text>
    
    <text x="90" y="850" class="small-text">🏢 トヨタ自動車：</text>
    <text x="90" y="865" class="step-text">• 技術文書：月1回夜間バッチ更新</text>
    <text x="90" y="880" class="step-text">• 品質データ：週1回定期更新</text>
    
    <text x="90" y="900" class="small-text">🏦 三菱UFJ銀行：</text>
    <text x="90" y="915" class="step-text">• 規制文書：週1回コンプライアンス確認後更新</text>
    
    <!-- MCP更新例 -->
    <rect x="400" y="810" width="300" height="120" rx="8" class="mcp-color" fill-opacity="0.1"/>
    <text x="550" y="830" text-anchor="middle" class="text">MCP更新の実装例</text>
    
    <text x="410" y="850" class="small-text">🚅 JR東日本：</text>
    <text x="410" y="865" class="step-text">• 運行情報：リアルタイム（秒単位）</text>
    <text x="410" y="880" class="step-text">• 駅混雑情報：5分間隔自動更新</text>
    
    <text x="410" y="900" class="small-text">🛒 楽天：</text>
    <text x="410" y="915" class="step-text">• 在庫情報：購入時点で即座更新</text>
    
    <!-- ハイブリッド例 -->
    <rect x="720" y="810" width="300" height="120" rx="8" class="process-color" fill-opacity="0.1"/>
    <text x="870" y="830" text-anchor="middle" class="text">ハイブリッド実装例</text>
    
    <text x="730" y="850" class="small-text">🏭 パナソニック：</text>
    <text x="730" y="865" class="step-text">• 技術文書（RAG）：月1回更新</text>
    <text x="730" y="880" class="step-text">• 生産データ（MCP）：リアルタイム</text>
    
    <text x="730" y="900" class="small-text">📱 ソフトバンク：</text>
    <text x="730" y="915" class="step-text">• 技術仕様（RAG）+ 通話状況（MCP）</text>
    
    <!-- 推奨パターン -->
    <rect x="1040" y="810" width="280" height="120" rx="8" class="success" fill-opacity="0.1"/>
    <text x="1180" y="830" text-anchor="middle" class="text">推奨実装パターン</text>
    
    <text x="1050" y="850" class="small-text">✅ 適材適所の使い分け：</text>
    <text x="1050" y="865" class="step-text">• 静的知識 → RAG（定期更新）</text>
    <text x="1050" y="880" class="step-text">• 動的データ → MCP（リアルタイム）</text>
    
    <text x="1050" y="900" class="small-text">⚡ 日本企業の成功要因：</text>
    <text x="1050" y="915" class="step-text">• 品質重視文化との適合性</text>
  </g>
  
  <!-- 更新頻度のタイムライン -->
  <g>
    <text x="700" y="970" text-anchor="middle" class="subtitle">更新頻度タイムライン比較</text>
    
    <!-- Timeline -->
    <line x1="100" y1="985" x2="1300" y2="985" stroke="#6c757d" stroke-width="3"/>
    
    <!-- RAG Timeline -->
    <text x="150" y="1000" text-anchor="middle" class="small-text">RAG更新</text>
    <circle cx="200" cy="985" r="8" class="rag-color"/>
    <circle cx="400" cy="985" r="8" class="rag-color"/>
    <circle cx="600" cy="985" r="8" class="rag-color"/>
    <circle cx="800" cy="985" r="8" class="rag-color"/>
    <text x="500" y="970" text-anchor="middle" class="step-text">月1回〜週1回の大規模更新</text>
    
    <!-- MCP Timeline -->
    <text x="1250" y="1000" text-anchor="middle" class="small-text">MCP更新</text>
    <rect x="950" y="980" width="300" height="10" rx="2" class="mcp-color" fill-opacity="0.6"/>
    <text x="1100" y="970" text-anchor="middle" class="step-text">連続的リアルタイム更新</text>
  </g>
</svg>
```
