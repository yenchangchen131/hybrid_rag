"""
Hybrid RAG 評估儀表板

Streamlit 前端應用程式

使用方式：
    uv run streamlit run app.py
"""

import json
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

from services import RAGService
from core.config import settings
from models.document import QueryModel


# 頁面配置
st.set_page_config(
    page_title="Hybrid RAG 評估儀表板",
    page_icon="🔍",
    layout="wide",
)

# 初始化 session state
if "rag_service" not in st.session_state:
    st.session_state.rag_service = None
if "results" not in st.session_state:
    st.session_state.results = {}  # {mode: results}
if "current_mode" not in st.session_state:
    st.session_state.current_mode = None


def load_queries() -> list[QueryModel]:
    """載入查詢"""
    with open(settings.queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [QueryModel(**q) for q in data]


def run_evaluation(queries: list[QueryModel], mode: str, top_k: int) -> list[dict]:
    """執行評估"""
    rag = st.session_state.rag_service
    if rag is None:
        rag = RAGService()
        st.session_state.rag_service = rag
    
    rag.initialize(mode=mode)
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, query in enumerate(queries):
        status_text.text(f"處理中: {i+1}/{len(queries)}")
        
        start_time = time.perf_counter()
        response = rag.answer(query.question, top_k=top_k, mode=mode)
        elapsed_time = time.perf_counter() - start_time
        
        gold_ids = set(query.gold_doc_ids)
        retrieved_ids = set(response.retrieved_doc_ids)
        hit_ids = gold_ids.intersection(retrieved_ids)
        
        result = {
            "question_id": query.question_id,
            "question": query.question,
            "question_type": query.question_type,
            "source_dataset": query.source_dataset,
            "gold_answer": query.gold_answer,
            "gold_doc_ids": query.gold_doc_ids,
            "generated_answer": response.answer,
            "retrieved_doc_ids": response.retrieved_doc_ids,
            "contexts": [
                {
                    "doc_id": ctx.doc_id,
                    "score": ctx.score,
                    "content": ctx.content,
                    "original_source": ctx.original_source,
                }
                for ctx in response.contexts
            ],
            "hit_count": len(hit_ids),
            "gold_count": len(gold_ids),
            "partial_hit": f"{len(hit_ids)}/{len(gold_ids)}",
            "is_hit": len(hit_ids) > 0,
            "response_time_ms": round(elapsed_time * 1000, 2),
        }
        results.append(result)
        progress_bar.progress((i + 1) / len(queries))
    
    status_text.text("✅ 完成!")
    return results


def calculate_metrics(results: list[dict]) -> dict:
    """計算指標"""
    total = len(results)
    
    # 整體統計
    total_hits = sum(1 for r in results if r["is_hit"])
    total_gold_docs = sum(r["gold_count"] for r in results)
    total_hit_docs = sum(r["hit_count"] for r in results)
    avg_time = sum(r["response_time_ms"] for r in results) / total if total > 0 else 0
    
    # 單一 gold doc 的 hit rate
    single_gold = [r for r in results if r["gold_count"] == 1]
    single_hits = sum(1 for r in single_gold if r["is_hit"])
    
    # MRR
    def calc_mrr(results):
        total_rr = 0
        for r in results:
            gold_ids = set(r["gold_doc_ids"])
            rr_sum = 0
            for gold_id in gold_ids:
                for rank, doc_id in enumerate(r["retrieved_doc_ids"], start=1):
                    if doc_id == gold_id:
                        rr_sum += 1.0 / rank
                        break
            total_rr += rr_sum / len(gold_ids) if gold_ids else 0
        return total_rr / len(results) if results else 0
    
    return {
        "total_questions": total,
        "hit_rate": total_hits / total if total > 0 else 0,
        "single_gold_hit_rate": single_hits / len(single_gold) if single_gold else 0,
        "partial_hit_rate": total_hit_docs / total_gold_docs if total_gold_docs > 0 else 0,
        "mrr": calc_mrr(results),
        "avg_response_time_ms": avg_time,
    }


def display_metrics_comparison():
    """顯示指標比較"""
    if not st.session_state.results:
        st.info("尚無評估結果，請先執行評估。")
        return
    
    # 計算各模式指標
    metrics_data = []
    for mode, results in st.session_state.results.items():
        metrics = calculate_metrics(results)
        metrics["mode"] = mode
        metrics_data.append(metrics)
    
    df = pd.DataFrame(metrics_data)
    df = df.set_index("mode")
    
    # 格式化顯示
    st.subheader("📊 指標比較")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("總題數", df["total_questions"].iloc[0])
    
    # 顯示各模式的指標
    for mode in df.index:
        st.markdown(f"### {mode.upper()}")
        cols = st.columns(5)
        cols[0].metric("Hit Rate", f"{df.loc[mode, 'hit_rate']:.2%}")
        cols[1].metric("Single Gold Hit Rate", f"{df.loc[mode, 'single_gold_hit_rate']:.2%}")
        cols[2].metric("Partial Hit Rate", f"{df.loc[mode, 'partial_hit_rate']:.2%}")
        cols[3].metric("MRR", f"{df.loc[mode, 'mrr']:.4f}")
        cols[4].metric("Avg Response Time", f"{df.loc[mode, 'avg_response_time_ms']:.0f} ms")


def display_results_table(mode: str):
    """顯示結果表格"""
    if mode not in st.session_state.results:
        return
    
    results = st.session_state.results[mode]
    
    # 轉換為 DataFrame
    df = pd.DataFrame([
        {
            "ID": r["question_id"][:8],
            "問題": r["question"][:50] + "..." if len(r["question"]) > 50 else r["question"],
            "類型": r["question_type"],
            "來源": r["source_dataset"],
            "命中": r["partial_hit"],
            "是否命中": "✅" if r["is_hit"] else "❌",
            "時間(ms)": r["response_time_ms"],
        }
        for r in results
    ])
    
    st.dataframe(df, use_container_width=True)


def display_question_detail(mode: str, question_idx: int):
    """顯示問題詳情"""
    if mode not in st.session_state.results:
        return
    
    result = st.session_state.results[mode][question_idx]
    
    st.markdown("---")
    st.subheader(f"📝 問題詳情")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**問題:**")
        st.write(result["question"])
        
        st.markdown("**標準答案:**")
        st.info(result["gold_answer"])
        
        st.markdown("**模型回答:**")
        st.success(result["generated_answer"])
    
    with col2:
        st.markdown("**Gold Doc IDs:**")
        for doc_id in result["gold_doc_ids"]:
            st.code(doc_id)
        
        st.markdown("**檢索結果:**")
        st.write(f"命中: {result['partial_hit']}")
        
    # 檢索到的文件
    st.markdown("### 📚 檢索到的文件")
    for i, ctx in enumerate(result["contexts"]):
        is_gold = ctx["doc_id"] in result["gold_doc_ids"]
        icon = "🎯" if is_gold else "📄"
        
        with st.expander(f"{icon} [{i+1}] {ctx['doc_id'][:16]}... (Score: {ctx['score']:.4f})"):
            st.markdown(f"**來源:** {ctx['original_source']}")
            st.markdown("**內容:**")
            st.write(ctx["content"])


def main():
    st.title("🔍 Hybrid RAG 評估儀表板")
    
    # 側邊欄
    with st.sidebar:
        st.header("⚙️ 設定")
        
        mode = st.selectbox(
            "檢索模式",
            ["hybrid", "vector", "keyword"],
            index=0,
        )
        
        top_k = st.slider("Top-K", min_value=1, max_value=20, value=5)
        
        if st.button("🚀 執行評估", type="primary", use_container_width=True):
            with st.spinner("載入查詢..."):
                queries = load_queries()
            
            st.info(f"正在以 {mode} 模式執行 {len(queries)} 題...")
            results = run_evaluation(queries, mode, top_k)
            st.session_state.results[mode] = results
            st.session_state.current_mode = mode
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 已完成的評估")
        for m in st.session_state.results.keys():
            st.write(f"✅ {m}")
    
    # 主區域
    tab1, tab2, tab3 = st.tabs(["📊 指標比較", "📋 結果列表", "🔎 問題詳情"])
    
    with tab1:
        display_metrics_comparison()
    
    with tab2:
        if st.session_state.results:
            selected_mode = st.selectbox(
                "選擇模式",
                list(st.session_state.results.keys()),
                key="results_mode"
            )
            display_results_table(selected_mode)
        else:
            st.info("請先執行評估。")
    
    with tab3:
        if st.session_state.results:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                selected_mode = st.selectbox(
                    "模式",
                    list(st.session_state.results.keys()),
                    key="detail_mode"
                )
                
                question_options = [
                    f"{i+1}. {r['question'][:30]}..."
                    for i, r in enumerate(st.session_state.results[selected_mode])
                ]
                selected_q = st.selectbox("選擇問題", question_options, key="detail_q")
                question_idx = question_options.index(selected_q)
            
            with col2:
                display_question_detail(selected_mode, question_idx)
        else:
            st.info("請先執行評估。")


if __name__ == "__main__":
    main()
