# Hybrid RAG System

混合式檢索增強生成系統 - 支援繁體中文問答，結合向量搜尋與關鍵字檢索。

## 🌟 核心特色

- **多模式檢索**: vector / keyword / hybrid (RRF Fusion)
- **完整評估系統**: 檢索指標 + LLM 語意評估
- **Streamlit 儀表板**: 視覺化比較、分組統計
- **多資料源**: DRCD、SQuAD、HotpotQA、2WikiMultiHopQA

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

| Tab | 功能 |
|-----|------|
| 📊 模式比較 | 三模式長條圖 + 完整指標表 |
| 📋 詳細報告 | 按資料來源/問題類型分組統計 |
| 📋 結果列表 | 各模式的問題列表與命中狀態 |
| 🔎 問題詳情 | Gold Docs 內容展開、檢索結果詳情 |

## 📊 CLI 評估

```bash
# 1. 批次執行
uv run python scripts/run_all_queries.py --mode hybrid

# 2. 計算檢索指標
uv run python scripts/calculate_metrics.py -i data/rag_results_hybrid.json

# 3. LLM 語意評估
uv run python scripts/evaluate_answers.py -i data/rag_results_hybrid.json
```

### CLI 輸出範例

```
==================================================
按資料來源分組
==================================================

【drcd】
  問題數:           20
  Hit Rate:         100.00%
  Partial Hit Rate: 100.00% (20/20)
  MRR:              0.9100

==================================================
總計
==================================================

  問題數:           60
  Hit Rate:         100.00%
  Partial Hit Rate: 83.96% (89/106)
  MRR:              0.6566
```

## 📈 評估指標

| 指標 | 說明 |
|------|------|
| Hit Rate | 單一 gold doc 問題的命中率 |
| Partial Hit Rate | 命中數/總 gold docs (如 89/106) |
| MRR | 平均 Reciprocal Rank（多 gold doc 取平均） |
| LLM Pass Rate | GPT-4o-mini 語意判斷通過率 |
| Response Time | 每題回應時間 (ms) |

## 📁 專案結構

```
hybrid_rag/
├── core/                    # 配置、資料庫、日誌
├── models/                  # Pydantic 資料模型
├── repositories/            # 資料庫操作層
├── services/                # 業務邏輯層
├── scripts/
│   ├── ingest_data.py
│   ├── run_all_queries.py
│   ├── calculate_metrics.py
│   └── evaluate_answers.py
├── data/
│   ├── corpus.json
│   ├── queries.json
│   └── rag_results_{mode}.json
├── app.py                   # Streamlit 儀表板
└── main.py                  # CLI 互動問答
```

## 🛠 開發

```bash
uv sync --extra dev   # 開發依賴
uv sync --extra ui    # Streamlit
```
