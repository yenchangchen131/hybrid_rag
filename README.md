# Hybrid RAG System for DRCD

這是一個專為 DRCD (Delta Reading Comprehension Dataset) 繁體中文閱讀理解資料集打造的 Hybrid RAG 系統。

## 🌟 核心特色 (Features)

*   **Hybrid Retrieval (混合檢索)**:
    *   **向量檢索 (Semantic Search)**: 使用 OpenAI `text-embedding-3-small` 模型配合 Cosine Similarity 計算語意相似度。
    *   **關鍵字檢索 (Keyword Search)**: 利用 MongoDB 內建的 `$text` 搜尋 (BM25) 捕捉精確關鍵字。
    *   **RRF Fusion**: 透過 Reciprocal Rank Fusion 演算法融合上述兩路檢索結果，達到最佳召回率。
*   **MongoDB 為核心**:
    *   所有資料（包含文本、Metadata、向量）皆儲存於 MongoDB。
    *   支援 Docker 部署或直接連線本地 MongoDB 服務。
*   **Streamlit 互動介面**:
    *   **Chatbot**: 類似 ChatGPT 的問答介面，並可展開查看檢索到的參考來源。
    *   **Evaluation Dashboard**: 內建評測看板，可一鍵執行驗證集迴歸測試，並視覺化 Recall@K 等指標。
*   **自動化評測**:
    *   包含 Recall, Precision, MRR 等指標的自動計算腳本。

## 📁 專案結構 (Structure)

```
.
├── docker-compose.yml      # MongoDB Docker 設定 (選用)
├── main.py                 # CLI 版本的問答入口
├── README.md               # 說明文件
├── data/                   # 資料存放區 (DRCD json, report等)
└── src/                    # 原始碼目錄
    ├── data_preprocess.py  # 下載與預處理 DRCD 資料
    ├── db_manager.py       # MongoDB 連線管理
    ├── ingest_data.py      # 資料匯入腳本
    ├── update_embeddings.py# 向量生成與更新腳本
    ├── drop_db.py          # 清除資料庫工具
    ├── embedding.py        # OpenAI Embedding 包裝
    ├── retriever.py        # Hybrid Retrieval 核心邏輯 (Vector + Keyword + RRF)
    ├── generator.py        # LLM 生成邏輯 (GPT-4o)
    ├── rag.py              # RAG 系統整合介面
    ├── streamlit_app.py    # 前端應用程式
    ├── evaluate.py         # 評測執行腳本
    └── calculate_metrics.py# 評測指標計算
```

## 🚀 快速開始 (Quick Start)

### 1. 環境設定

確認已安裝 `uv` 套件管理工具與 Python 3.10+。
複製 `.env.example` 為 `.env` 並填入您的 OpenAI API Key：
```bash
OPENAI_API_KEY=sk-xxxxxx
```

安裝相依套件：
```bash
uv sync
```

### 2. 資料庫準備

確保 MongoDB 正在運行 (Port 27017)。若無本機 MongoDB，可使用 Docker：
```bash
docker-compose up -d
```

### 3. 資料初始化 (Data Ingestion)

首次執行需依序跑過資料處理流程：

```bash
# 1. 下載並處理資料
uv run src/data_preprocess.py

# 2. 匯入文字資料到 MongoDB
uv run src/ingest_data.py

# 3. 生成向量 (需要 OpenAI API，需時數分鐘)
uv run src/update_embeddings.py
```

### 4. 啟動應用程式

**Web 介面 (推薦)**:
包含問答機器人與評測看板。
```bash
uv run streamlit run src/streamlit_app.py
```

**CLI 介面**:
```bash
uv run main.py
```

## 📊 評測結果 (Performance)

本系統在 50 題 DRCD 驗證集上的表現：

| Metric | Score | 說明 |
| :--- | :--- | :--- |
| **Recall@5** | **98.0%** | 前 5 筆結果中，有 98% 的機率包含正確答案段落。 |
| **Recall@1** | 70.0% | 第 1 筆結果即為正確答案的機率。 |
| **MRR@5** | 0.812 | 平均倒數排名，顯示正確答案通常排在極前面。 |

## 🛠 工具指令

*   **重跑評測**: `uv run src/evaluate.py`
*   **計算指標**: `uv run src/calculate_metrics.py`
*   **刪除資料庫**: `uv run src/drop_db.py`
