# Hybrid RAG System

混合式檢索增強生成系統 - 支援繁體中文問答，結合向量搜尋與關鍵字檢索。

## 🌟 核心特色

- **多模式檢索**:
  - `vector`: 純向量檢索 (OpenAI Embeddings + Cosine Similarity)
  - `keyword`: 純關鍵字檢索 (MongoDB Text Search)
  - `hybrid`: 混合檢索 (Vector + Keyword + RRF Fusion)
- **完整評估系統**: 檢索指標 + LLM 語意評估
- **Streamlit 儀表板**: 視覺化比較各模式效能
- **多資料源**: DRCD、SQuAD、HotpotQA、2WikiMultiHopQA

## 📁 專案結構

```
hybrid_rag/
├── core/                    # 配置、資料庫、日誌
├── models/                  # Pydantic 資料模型
├── repositories/            # 資料庫操作層
├── services/                # 業務邏輯層
├── scripts/
│   ├── ingest_data.py           # 資料導入
│   ├── run_query.py             # 單一查詢
│   ├── run_all_queries.py       # 批次查詢（支援模式選擇）
│   ├── calculate_metrics.py     # 計算檢索指標
│   └── evaluate_answers.py      # LLM 語意評估
├── data/
│   ├── corpus.json
│   └── queries.json
├── app.py                   # Streamlit 儀表板
└── main.py                  # CLI 互動問答
```

## 🚀 快速開始

```bash
cp .env.example .env       # 填入 OPENAI_API_KEY
uv sync
docker compose up -d mongodb
uv run python scripts/ingest_data.py
```

## 🖥️ Streamlit 儀表板

```bash
uv run streamlit run app.py
```

**功能：**
- 選擇模式 (hybrid/vector/keyword) 執行評估
- 指標比較視覺化
- 展開查看單一問題詳情

## 📊 批次評估

### 步驟 1：執行查詢

```bash
uv run python scripts/run_all_queries.py --mode hybrid
uv run python scripts/run_all_queries.py --mode vector
uv run python scripts/run_all_queries.py --mode keyword
```

輸出：`data/rag_results_{mode}.json`（包含每題 `response_time_ms`）

### 步驟 2：計算檢索指標

```bash
uv run python scripts/calculate_metrics.py --input data/rag_results_hybrid.json
```

輸出：`data/evaluation_metrics_{mode}.json`

### 步驟 3：LLM 語意評估（選用）

```bash
uv run python scripts/evaluate_answers.py --input data/rag_results_hybrid.json
```

輸出：`data/answer_evaluation_{mode}.json`

## 📈 評估指標

| 指標 | 說明 |
|------|------|
| **Hit Rate** | 單一 gold doc 問題的命中率 |
| **Partial Hit Rate** | 命中的 gold docs / 總 gold docs (如 2/5) |
| **MRR** | 平均 Reciprocal Rank（多 gold doc 取平均） |
| **Pass Rate** | LLM 判斷語意一致的比例 |
| **Response Time** | 每題回應時間 (ms) |

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
uv sync --extra ui    # Streamlit
uv run pytest
```

## 🐳 Docker

```bash
docker compose up -d mongodb
docker build -t hybrid-rag .
docker run --env-file .env hybrid-rag
```
