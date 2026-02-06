# Hybrid RAG System

混合式檢索增強生成系統 - 支援繁體中文問答，結合向量搜尋與關鍵字檢索。

## 🌟 核心特色

- **多模式檢索**: vector / keyword / hybrid
- **完整評估系統**: 檢索指標 + LLM 語意評估
- **Streamlit 儀表板**: 視覺化比較各模式效能
- **多資料源**: DRCD、SQuAD、HotpotQA、2WikiMultiHopQA

## 📁 專案結構

```
hybrid_rag/
├── core/                        # 配置、資料庫、日誌
├── models/                      # Pydantic 資料模型
├── repositories/                # 資料庫操作層
├── services/                    # 業務邏輯層
│   ├── retrieval_service.py     # 檢索服務（支援多模式）
│   ├── generation_service.py    # 生成服務
│   └── rag_service.py           # RAG 整合服務
├── scripts/
│   ├── ingest_data.py           # 資料導入
│   ├── run_query.py             # 單一查詢
│   ├── run_all_queries.py       # 批次查詢
│   ├── calculate_metrics.py     # 計算檢索指標
│   └── evaluate_answers.py      # LLM 語意評估
├── data/
│   ├── corpus.json              # 文件語料庫
│   ├── queries.json             # 測試問題集
│   ├── rag_results_{mode}.json  # 各模式 RAG 結果
│   ├── evaluation_metrics_{mode}.json
│   └── answer_evaluation_{mode}.json
├── app.py                       # Streamlit 儀表板
└── main.py                      # CLI 互動問答
```

## 🚀 快速開始

```bash
cp .env.example .env       # 填入 OPENAI_API_KEY
uv sync --extra ui
docker compose up -d mongodb
uv run python scripts/ingest_data.py
```

## 🖥️ Streamlit 儀表板

```bash
uv run streamlit run app.py
```

| 功能 | 說明 |
|------|------|
| 模式選擇 | hybrid / vector / keyword |
| 批次評估 | 執行 50 題並儲存結果 |
| LLM 評估 | GPT-4o-mini 判斷答案正確性 |
| 指標比較 | 三模式長條圖比較 |
| 問題詳情 | 展開查看 Gold Docs 內容 |

## 📊 CLI 評估流程

```bash
# 1. 批次執行（自動儲存 response_time_ms）
uv run python scripts/run_all_queries.py --mode hybrid
uv run python scripts/run_all_queries.py --mode vector
uv run python scripts/run_all_queries.py --mode keyword

# 2. 計算檢索指標
uv run python scripts/calculate_metrics.py -i data/rag_results_hybrid.json

# 3. LLM 語意評估（選用）
uv run python scripts/evaluate_answers.py -i data/rag_results_hybrid.json
```

## 📈 評估指標

| 指標 | 說明 |
|------|------|
| Hit Rate | 單一 gold doc 問題的命中率 |
| Partial Hit Rate | 命中的 gold docs / 總 gold docs |
| MRR | 平均 Reciprocal Rank（多 gold doc 取平均） |
| LLM Pass Rate | GPT-4o-mini 判斷語意一致的比例 |
| Response Time | 每題回應時間 (ms) |

## 🛠 開發

```bash
uv sync --extra dev   # 開發依賴
uv sync --extra api   # FastAPI
uv sync --extra ui    # Streamlit
```
