# Hybrid RAG System

混合式檢索增強生成系統 - 支援繁體中文問答，結合向量搜尋與關鍵字檢索。

## 🌟 核心特色

- **多模式檢索**:
  - `vector`: 純向量檢索 (OpenAI Embeddings + Cosine Similarity)
  - `keyword`: 純關鍵字檢索 (MongoDB Text Search)
  - `hybrid`: 混合檢索 (Vector + Keyword + RRF Fusion)
- **分層架構**: Core / Models / Repositories / Services
- **多資料源**: DRCD、SQuAD、HotpotQA、2WikiMultiHopQA

## 📁 專案結構

```
hybrid_rag/
├── core/                # 配置、資料庫、日誌
├── models/              # Pydantic 資料模型
├── repositories/        # 資料庫操作層
├── services/            # 業務邏輯層
├── scripts/             # CLI 腳本
│   ├── ingest_data.py       # 資料導入
│   ├── run_query.py         # 單一查詢
│   ├── run_all_queries.py   # 批次查詢
│   └── calculate_metrics.py # 計算指標
├── data/
│   ├── corpus.json
│   └── queries.json
└── main.py
```

## 🚀 快速開始

```bash
cp .env.example .env       # 填入 OPENAI_API_KEY
uv sync
docker compose up -d mongodb
uv run python scripts/ingest_data.py
uv run python main.py
```

## 📊 批次評估

### 步驟 1：執行查詢（選擇模式）

```bash
# 混合檢索 (預設)
uv run python scripts/run_all_queries.py --mode hybrid

# 純向量檢索
uv run python scripts/run_all_queries.py --mode vector

# 純關鍵字檢索
uv run python scripts/run_all_queries.py --mode keyword
```

輸出：`data/rag_results_{mode}.json`

### 步驟 2：計算指標

```bash
uv run python scripts/calculate_metrics.py --input data/rag_results_hybrid.json
```

輸出：`data/evaluation_metrics.json`

### 指標說明

| 指標 | 說明 |
|------|------|
| **Hit Rate** | 單一 gold doc 問題的命中率 |
| **Partial Hit Rate** | 命中 gold docs / 總 gold docs (如 2/5) |
| **MRR** | 平均 Reciprocal Rank，衡量排序精準度 |

## 📋 資料格式

**corpus.json**
```json
{"doc_id": "uuid", "content": "...", "original_source": "drcd", "is_gold": false}
```

**queries.json**
```json
{"question_id": "uuid", "question": "...", "gold_doc_ids": ["id1", "id2"]}
```

## 🛠 開發

```bash
uv sync --extra dev   # 開發依賴
uv sync --extra api   # FastAPI
uv run pytest
```
