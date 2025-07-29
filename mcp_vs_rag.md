<img width="967" height="606" alt="image" src="https://github.com/user-attachments/assets/33542cc2-710b-4d98-ab48-cbaee7268814" />

```
<svg viewBox="0 0 1200 800" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .title { font-family: "Noto Sans JP", Arial, sans-serif; font-size: 24px; font-weight: bold; fill: #2c3e50; }
      .subtitle { font-family: "Noto Sans JP", Arial, sans-serif; font-size: 18px; font-weight: bold; fill: #34495e; }
      .text { font-family: "Noto Sans JP", Arial, sans-serif; font-size: 14px; fill: #2c3e50; }
      .small-text { font-family: "Noto Sans JP", Arial, sans-serif; font-size: 12px; fill: #7f8c8d; }
      .tiny-text { font-family: "Noto Sans JP", Arial, sans-serif; font-size: 10px; fill: #95a5a6; }
      .rag-color { fill: #3498db; stroke: #2980b9; }
      .mcp-color { fill: #e74c3c; stroke: #c0392b; }
      .data-color { fill: #f39c12; stroke: #e67e22; }
      .fresh-color { fill: #2ecc71; stroke: #27ae60; }
      .old-color { fill: #95a5a6; stroke: #7f8c8d; }
      .arrow { stroke: #34495e; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }
      .dashed { stroke-dasharray: 8,4; }
      .dotted { stroke-dasharray: 2,3; }
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#34495e"/>
    </marker>
  </defs>
  
  <!-- Main Title -->
  <text x="600" y="30" text-anchor="middle" class="title">RAG vs MCP：データソースの根本的な違い</text>
  
  <!-- RAG Section -->
  <g>
    <rect x="50" y="60" width="500" height="320" rx="10" class="rag-color" fill-opacity="0.1" stroke-width="2"/>
    <text x="300" y="85" text-anchor="middle" class="subtitle">RAG - 図書館モデル（としょかんモデル）</text>
    
    <!-- Library Icon -->
    <rect x="80" y="100" width="60" height="50" rx="5" fill="#8e44ad" stroke="#7b68ee"/>
    <rect x="85" y="105" width="10" height="30" fill="white"/>
    <rect x="100" y="105" width="10" height="30" fill="white"/>
    <rect x="115" y="105" width="10" height="30" fill="white"/>
    <text x="110" y="170" text-anchor="middle" class="small-text">蔵書</text>
    
    <!-- Data Processing Flow -->
    <text x="80" y="195" class="text">データの前処理プロセス：</text>
    
    <!-- Original Documents -->
    <rect x="80" y="210" width="80" height="30" rx="3" class="data-color"/>
    <text x="120" y="230" text-anchor="middle" class="small-text">原文書</text>
    
    <!-- Arrow 1 -->
    <path d="M 160 225 L 190 225" class="arrow"/>
    
    <!-- Chunking -->
    <rect x="190" y="210" width="80" height="30" rx="3" class="rag-color"/>
    <text x="230" y="230" text-anchor="middle" class="small-text">分割処理</text>
    
    <!-- Arrow 2 -->
    <path d="M 270 225 L 300 225" class="arrow"/>
    
    <!-- Embedding -->
    <rect x="300" y="210" width="80" height="30" rx="3" class="rag-color"/>
    <text x="340" y="230" text-anchor="middle" class="small-text">ベクトル化</text>
    
    <!-- Arrow 3 -->
    <path d="M 380 225 L 410 225" class="arrow"/>
    
    <!-- Vector DB -->
    <rect x="410" y="210" width="80" height="30" rx="3" class="rag-color"/>
    <text x="450" y="230" text-anchor="middle" class="small-text">ベクトルDB</text>
    
    <!-- Search Process -->
    <text x="80" y="270" class="text">検索プロセス：</text>
    
    <!-- User Query -->
    <ellipse cx="120" cy="295" rx="40" ry="15" fill="#ecf0f1" stroke="#bdc3c7"/>
    <text x="120" y="300" text-anchor="middle" class="small-text">ユーザー質問</text>
    
    <!-- Similarity Search -->
    <path d="M 160 295 L 220 295" class="arrow dashed"/>
    <text x="190" y="290" text-anchor="middle" class="tiny-text">類似度計算</text>
    
    <!-- Retrieved Docs -->
    <rect x="220" y="285" width="80" height="20" rx="3" class="old-color"/>
    <text x="260" y="298" text-anchor="middle" class="small-text">関連文書片</text>
    
    <!-- Characteristics -->
    <text x="80" y="330" class="text">特徴：</text>
    <text x="90" y="345" class="small-text">• 静的知識ベース（過去のスナップショット）</text>
    <text x="90" y="360" class="small-text">• 推測的検索（類似度ベース）</text>
    <text x="320" y="345" class="small-text">• バッチ処理が必要</text>
    <text x="320" y="360" class="small-text">• 更新コストが高い</text>
  </g>
  
  <!-- MCP Section -->
  <g>
    <rect x="650" y="60" width="500" height="320" rx="10" class="mcp-color" fill-opacity="0.1" stroke-width="2"/>
    <text x="900" y="85" text-anchor="middle" class="subtitle">MCP - サービスデスクモデル</text>
    
    <!-- Service Desk Icon -->
    <circle cx="720" cy="125" r="25" fill="#27ae60" stroke="#2ecc71"/>
    <rect x="710" y="120" width="20" height="3" fill="white"/>
    <rect x="710" y="125" width="20" height="3" fill="white"/>
    <rect x="710" y="130" width="20" height="3" fill="white"/>
    <text x="720" y="170" text-anchor="middle" class="small-text">リアルタイム</text>
    
    <!-- Real-time Data Sources -->
    <text x="680" y="195" class="text">リアルタイムデータソース：</text>
    
    <!-- API Services -->
    <rect x="680" y="210" width="70" height="25" rx="3" class="fresh-color"/>
    <text x="715" y="227" text-anchor="middle" class="small-text">API</text>
    
    <rect x="760" y="210" width="70" height="25" rx="3" class="fresh-color"/>
    <text x="795" y="227" text-anchor="middle" class="small-text">データベース</text>
    
    <rect x="840" y="210" width="70" height="25" rx="3" class="fresh-color"/>
    <text x="875" y="227" text-anchor="middle" class="small-text">ファイル</text>
    
    <rect x="920" y="210" width="70" height="25" rx="3" class="fresh-color"/>
    <text x="955" y="227" text-anchor="middle" class="small-text">Webサービス</text>
    
    <!-- Direct Access Process -->
    <text x="680" y="260" class="text">直接アクセスプロセス：</text>
    
    <!-- User Request -->
    <ellipse cx="720" cy="285" rx="40" ry="15" fill="#ecf0f1" stroke="#bdc3c7"/>
    <text x="720" y="290" text-anchor="middle" class="small-text">ユーザー要求</text>
    
    <!-- Direct Call -->
    <path d="M 760 285 L 820 285" class="arrow"/>
    <text x="790" y="280" text-anchor="middle" class="tiny-text">直接呼び出し</text>
    
    <!-- Live Data -->
    <rect x="820" y="275" width="80" height="20" rx="3" class="fresh-color"/>
    <text x="860" y="288" text-anchor="middle" class="small-text">最新データ</text>
    
    <!-- Characteristics -->
    <text x="680" y="320" class="text">特徴：</text>
    <text x="690" y="335" class="small-text">• 動的データソース（リアルタイム）</text>
    <text x="690" y="350" class="small-text">• 精密な呼び出し（パラメータベース）</text>
    <text x="920" y="335" class="small-text">• オンデマンド取得</text>
    <text x="920" y="350" class="small-text">• 更新コストが低い</text>
  </g>
  
  <!-- Comparison Section -->
  <g>
    <text x="600" y="420" text-anchor="middle" class="subtitle">データの新鮮度と精度の比較</text>
    
    <!-- Timeline -->
    <line x1="100" y1="460" x2="1100" y2="460" stroke="#34495e" stroke-width="2"/>
    
    <!-- Time markers -->
    <text x="200" y="450" text-anchor="middle" class="small-text">過去</text>
    <text x="600" y="450" text-anchor="middle" class="small-text">現在</text>
    <text x="1000" y="450" text-anchor="middle" class="small-text">未来</text>
    
    <!-- RAG Data Age -->
    <rect x="150" y="470" width="300" height="30" rx="5" class="old-color" fill-opacity="0.6"/>
    <text x="300" y="490" text-anchor="middle" class="text">RAGデータ範囲（古いデータ）</text>
    
    <!-- MCP Data Age -->
    <circle cx="600" cy="485" r="15" class="fresh-color"/>
    <text x="600" y="520" text-anchor="middle" class="text">MCPデータポイント（最新）</text>
    
    <!-- Accuracy Comparison -->
    <text x="600" y="560" text-anchor="middle" class="subtitle">検索精度の違い</text>
    
    <!-- RAG Accuracy -->
    <g>
      <text x="300" y="590" text-anchor="middle" class="text">RAG検索精度</text>
      <rect x="200" y="600" width="200" height="20" rx="10" fill="#ecf0f1" stroke="#bdc3c7"/>
      <rect x="200" y="600" width="140" height="20" rx="10" class="rag-color" fill-opacity="0.7"/>
      <text x="300" y="635" text-anchor="middle" class="small-text">70% - 類似度ベース</text>
    </g>
    
    <!-- MCP Accuracy -->
    <g>
      <text x="900" y="590" text-anchor="middle" class="text">MCP取得精度</text>
      <rect x="800" y="600" width="200" height="20" rx="10" fill="#ecf0f1" stroke="#bdc3c7"/>
      <rect x="800" y="600" width="190" height="20" rx="10" class="fresh-color" fill-opacity="0.7"/>
      <text x="900" y="635" text-anchor="middle" class="small-text">95% - パラメータベース</text>
    </g>
  </g>
  
  <!-- Examples Section -->
  <g>
    <text x="600" y="680" text-anchor="middle" class="subtitle">具体例（ぐたいれい）</text>
    
    <!-- RAG Example -->
    <rect x="50" y="700" width="500" height="80" rx="5" class="rag-color" fill-opacity="0.1"/>
    <text x="70" y="720" class="text">RAG例：「SSL証明書の設定方法は？」</text>
    <text x="70" y="740" class="small-text">→ 過去の技術文書から類似内容を検索</text>
    <text x="70" y="755" class="small-text">→ 古い設定方法が返される可能性</text>
    <text x="70" y="770" class="tiny-text">データソース：静的文書アーカイブ</text>
    
    <!-- MCP Example -->
    <rect x="650" y="700" width="500" height="80" rx="5" class="mcp-color" fill-opacity="0.1"/>
    <text x="670" y="720" class="text">MCP例：「現在のSSL証明書の有効期限は？」</text>
    <text x="670" y="740" class="small-text">→ 証明書管理APIを直接呼び出し</text>
    <text x="670" y="755" class="small-text">→ リアルタイムの正確な期限を取得</text>
    <text x="670" y="770" class="tiny-text">データソース：ライブシステムAPI</text>
  </g>
</svg>
```
