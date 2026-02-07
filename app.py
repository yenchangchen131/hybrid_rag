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

DATA_DIR = Path("data")
MODES = ["hybrid", "vector", "keyword"]


# ===================== 資料管理 =====================

def get_result_path(mode: str) -> Path:
    return DATA_DIR / f"rag_results_{mode}.json"


def get_metrics_path(mode: str) -> Path:
    return DATA_DIR / f"evaluation_metrics_{mode}.json"


def get_answer_eval_path(mode: str) -> Path:
    return DATA_DIR / f"answer_evaluation_{mode}.json"


def load_existing_results() -> dict:
    """載入已存在的結果檔案（包含 LLM 評估）"""
    results = {}
    
    for mode in MODES:
        path = get_result_path(mode)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            mode_results = data.get("results", data)
            
            # 嘗試載入 LLM 評估結果並合併
            eval_path = get_answer_eval_path(mode)
            if eval_path.exists():
                with open(eval_path, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                eval_results = eval_data.get("results", [])
                
                # 建立 question_id -> eval_result 的映射
                eval_map = {e["question_id"]: e for e in eval_results}
                
                # 合併 LLM 評估資料
                for r in mode_results:
                    q_id = r.get("question_id")
                    if q_id and q_id in eval_map:
                        r["llm_judgment"] = eval_map[q_id].get("llm_judgment")
                        r["is_pass"] = eval_map[q_id].get("is_pass", False)
            
            results[mode] = mode_results
    
    return results


def save_results(mode: str, results: list[dict], metadata: dict = None):
    """儲存結果到檔案"""
    output_data = {
        "metadata": metadata or {},
        "results": results,
    }
    path = get_result_path(mode)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def load_queries() -> list[QueryModel]:
    """載入查詢"""
    with open(settings.queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [QueryModel(**q) for q in data]


# ===================== 評估執行 =====================

def run_evaluation(queries: list[QueryModel], mode: str, top_k: int) -> list[dict]:
    """執行評估"""
    if "rag_service" not in st.session_state:
        st.session_state.rag_service = RAGService()
    
    rag = st.session_state.rag_service
    rag.initialize(mode=mode)
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_time = 0.0
    
    for i, query in enumerate(queries):
        status_text.text(f"處理中: {i+1}/{len(queries)}")
        
        start_time = time.perf_counter()
        response = rag.answer(query.question, top_k=top_k, mode=mode)
        elapsed_time = time.perf_counter() - start_time
        total_time += elapsed_time
        
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
    
    # 儲存結果
    metadata = {
        "mode": mode,
        "top_k": top_k,
        "total_questions": len(results),
        "total_time_seconds": round(total_time, 2),
        "avg_response_time_ms": round((total_time / len(results)) * 1000, 2),
        "timestamp": datetime.now().isoformat(),
    }
    save_results(mode, results, metadata)
    
    return results


# ===================== 指標計算 =====================

def calculate_metrics(results: list[dict]) -> dict:
    """計算檢索與生成指標"""
    total = len(results)
    if total == 0:
        return {}
    
    # 預處理：計算缺失的欄位
    for r in results:
        gold_ids = set(r.get("gold_doc_ids", []))
        retrieved_ids = set(r.get("retrieved_doc_ids", []))
        hit_ids = gold_ids.intersection(retrieved_ids)
        
        if "hit_count" not in r:
            r["hit_count"] = len(hit_ids)
        if "gold_count" not in r:
            r["gold_count"] = len(gold_ids)
        if "is_hit" not in r:
            r["is_hit"] = len(hit_ids) > 0
    
    # 檢索指標
    total_hits = sum(1 for r in results if r.get("is_hit", False))
    total_gold_docs = sum(r.get("gold_count", 0) for r in results)
    total_hit_docs = sum(r.get("hit_count", 0) for r in results)
    avg_time = sum(r.get("response_time_ms", 0) for r in results) / total
    
    # 單一 gold doc 的 hit rate
    single_gold = [r for r in results if r.get("gold_count", 0) == 1]
    single_hits = sum(1 for r in single_gold if r.get("is_hit", False))
    
    # MRR（平均 RR）
    def calc_mrr(results):
        total_rr = 0
        for r in results:
            gold_ids = set(r.get("gold_doc_ids", []))
            retrieved_ids = r.get("retrieved_doc_ids", [])
            rr_sum = 0
            for gold_id in gold_ids:
                for rank, doc_id in enumerate(retrieved_ids, start=1):
                    if doc_id == gold_id:
                        rr_sum += 1.0 / rank
                        break
            total_rr += rr_sum / len(gold_ids) if gold_ids else 0
        return total_rr / len(results) if results else 0
    
    # 生成指標（如果有 LLM 評估結果）
    has_llm_eval = any("is_pass" in r for r in results)
    if has_llm_eval:
        passed = sum(1 for r in results if r.get("is_pass", False))
        pass_rate = passed / total
    else:
        passed = None
        pass_rate = None
    
    return {
        "total_questions": total,
        "hit_rate": total_hits / total,
        "single_gold_hit_rate": single_hits / len(single_gold) if single_gold else 0,
        "partial_hit_rate": total_hit_docs / total_gold_docs if total_gold_docs > 0 else 0,
        "mrr": calc_mrr(results),
        "avg_response_time_ms": avg_time,
        # 生成指標
        "llm_passed": passed,
        "llm_pass_rate": pass_rate,
    }


# ===================== LLM 語意評估 =====================

def run_llm_evaluation(results: list[dict], mode: str) -> list[dict]:
    """執行 LLM 語意評估"""
    from openai import OpenAI
    
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    PROMPT = """請判斷「模型回答」是否與「標準答案」語意一致。

問題：{question}
標準答案：{gold_answer}
模型回答：{model_answer}

判斷標準：
- 如果模型回答包含標準答案的核心資訊，且沒有明顯錯誤，請回答 "Pass"
- 如果模型回答與標準答案語意不一致、有錯誤、或完全無關，請回答 "Fail"

請只回答 "Pass" 或 "Fail"。"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, r in enumerate(results):
        status_text.text(f"語意評估中: {i+1}/{len(results)}")
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": PROMPT.format(
                    question=r["question"],
                    gold_answer=r.get("gold_answer", ""),
                    model_answer=r.get("generated_answer", ""),
                )}],
                temperature=0,
                max_tokens=10,
            )
            raw = response.choices[0].message.content.strip()
            r["llm_judgment"] = raw
            r["is_pass"] = raw.lower() == "pass"
        except Exception as e:
            r["llm_judgment"] = f"Error: {e}"
            r["is_pass"] = False
        
        progress_bar.progress((i + 1) / len(results))
    
    status_text.text("✅ 語意評估完成!")
    
    # 更新儲存
    save_results(mode, results)
    
    return results


# ===================== UI 元件 =====================

def calculate_grouped_metrics(results: list[dict]) -> dict:
    """計算分組指標"""
    from collections import defaultdict
    
    # 預處理
    for r in results:
        gold_ids = set(r.get("gold_doc_ids", []))
        retrieved_ids = set(r.get("retrieved_doc_ids", []))
        hit_ids = gold_ids.intersection(retrieved_ids)
        r["hit_count"] = len(hit_ids)
        r["gold_count"] = len(gold_ids)
        r["is_hit"] = len(hit_ids) > 0
    
    def calc_mrr(subset):
        total_rr = 0
        for r in subset:
            gold_ids = set(r.get("gold_doc_ids", []))
            retrieved_ids = r.get("retrieved_doc_ids", [])
            rr_sum = 0
            for gold_id in gold_ids:
                for rank, doc_id in enumerate(retrieved_ids, start=1):
                    if doc_id == gold_id:
                        rr_sum += 1.0 / rank
                        break
            total_rr += rr_sum / len(gold_ids) if gold_ids else 0
        return total_rr / len(subset) if subset else 0
    
    def calc_group(subset):
        total = len(subset)
        gold_docs = sum(r["gold_count"] for r in subset)
        hit_docs = sum(r["hit_count"] for r in subset)
        single_gold = [r for r in subset if r["gold_count"] == 1]
        single_hits = sum(1 for r in single_gold if r["is_hit"])
        return {
            "total": total,
            "gold_docs": gold_docs,
            "hit_docs": hit_docs,
            "partial_hit_rate": hit_docs / gold_docs if gold_docs > 0 else 0,
            "hit_rate": single_hits / len(single_gold) if single_gold else None,
            "mrr": calc_mrr(subset),
        }
    
    # 按資料來源分組
    by_source = defaultdict(list)
    for r in results:
        by_source[r.get("source_dataset", "unknown")].append(r)
    
    # 按問題類型分組
    by_type = defaultdict(list)
    for r in results:
        by_type[r.get("question_type", "unknown")].append(r)
    
    return {
        "by_source": {k: calc_group(v) for k, v in by_source.items()},
        "by_type": {k: calc_group(v) for k, v in by_type.items()},
        "total": calc_group(results),
    }


def display_metrics_comparison(all_results: dict):
    """顯示指標比較（三模式並排）"""
    if not all_results:
        st.info("尚無評估結果。請執行評估或確認 data 目錄中有結果檔案。")
        return
    
    # 選擇模式
    available_modes = [m for m in MODES if m in all_results]
    
    # Tab: 比較圖表 vs 詳細報告
    sub_tab1, sub_tab2 = st.tabs(["📊 模式比較", "📋 詳細報告"])
    
    with sub_tab1:
        st.subheader("📊 三模式指標比較")
        
        # 計算各模式指標
        metrics_list = []
        for mode in MODES:
            if mode in all_results:
                m = calculate_metrics(all_results[mode])
                m["mode"] = mode
                metrics_list.append(m)
        
        if not metrics_list:
            return
        
        df = pd.DataFrame(metrics_list).set_index("mode")
        
        # 指標選擇
        metric_options = {
            "Hit Rate": "hit_rate",
            "Partial Hit Rate": "partial_hit_rate",
            "MRR": "mrr",
            "Avg Response Time (ms)": "avg_response_time_ms",
        }
        if df["llm_pass_rate"].notna().any():
            metric_options["LLM Pass Rate"] = "llm_pass_rate"
        
        selected_metric = st.selectbox("選擇指標", list(metric_options.keys()))
        metric_col = metric_options[selected_metric]
        
        # 長條圖
        chart_data = df[[metric_col]].dropna()
        chart_data.columns = [selected_metric]
        st.bar_chart(chart_data)
        
        # 完整表格
        st.markdown("### 完整指標表")
        display_df = df[["hit_rate", "partial_hit_rate", "mrr", "avg_response_time_ms"]].copy()
        display_df.columns = ["Hit Rate", "Partial HR", "MRR", "Avg Time (ms)"]
        for col in ["Hit Rate", "Partial HR"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
        display_df["MRR"] = display_df["MRR"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
        display_df["Avg Time (ms)"] = display_df["Avg Time (ms)"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "-")
        if "llm_pass_rate" in df.columns:
            display_df["LLM Pass Rate"] = df["llm_pass_rate"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
        st.dataframe(display_df.T, width="stretch")
    
    with sub_tab2:
        selected_mode = st.selectbox("選擇模式查看詳細", available_modes, key="detail_metrics_mode")
        results = all_results[selected_mode]
        grouped = calculate_grouped_metrics(results)
        
        # 按資料來源
        st.markdown("### 📚 按資料來源分組")
        source_data = []
        for source, stats in grouped["by_source"].items():
            source_data.append({
                "來源": source,
                "問題數": stats["total"],
                "Hit Rate": f"{stats['hit_rate']:.2%}" if stats['hit_rate'] else "-",
                "Partial HR": f"{stats['partial_hit_rate']:.2%} ({stats['hit_docs']}/{stats['gold_docs']})",
                "MRR": f"{stats['mrr']:.4f}",
            })
        st.dataframe(pd.DataFrame(source_data), width="stretch", hide_index=True)
        
        # 按問題類型
        st.markdown("### 📈 按問題類型分組")
        type_data = []
        for q_type, stats in grouped["by_type"].items():
            type_data.append({
                "類型": q_type,
                "問題數": stats["total"],
                "Hit Rate": f"{stats['hit_rate']:.2%}" if stats['hit_rate'] else "-",
                "Partial HR": f"{stats['partial_hit_rate']:.2%} ({stats['hit_docs']}/{stats['gold_docs']})",
                "MRR": f"{stats['mrr']:.4f}",
            })
        st.dataframe(pd.DataFrame(type_data), width="stretch", hide_index=True)
        
        # 總計
        st.markdown("### 📊 總計")
        total = grouped["total"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("問題數", total["total"])
        col2.metric("Hit Rate", f"{total['hit_rate']:.2%}" if total['hit_rate'] else "-")
        col3.metric("Partial HR", f"{total['partial_hit_rate']:.2%} ({total['hit_docs']}/{total['gold_docs']})")
        col4.metric("MRR", f"{total['mrr']:.4f}")


def display_results_table(mode: str, results: list[dict]):
    """顯示結果表格"""
    # 預處理欄位
    for r in results:
        if "partial_hit" not in r:
            gold_ids = set(r.get("gold_doc_ids", []))
            retrieved_ids = set(r.get("retrieved_doc_ids", []))
            hit_count = len(gold_ids.intersection(retrieved_ids))
            r["partial_hit"] = f"{hit_count}/{len(gold_ids)}"
            r["is_hit"] = hit_count > 0
    
    df = pd.DataFrame([
        {
            "ID": r.get("question_id", "")[:8],
            "問題": (r.get("question", "")[:40] + "...") if len(r.get("question", "")) > 40 else r.get("question", ""),
            "類型": r.get("question_type", "-"),
            "來源": r.get("source_dataset", "-"),
            "命中": r.get("partial_hit", "-"),
            "Hit": "✅" if r.get("is_hit") else "❌",
            "LLM": "✅" if r.get("is_pass") else ("❌" if "is_pass" in r else "-"),
            "Time(ms)": r.get("response_time_ms", 0),
        }
        for r in results
    ])
    
    st.dataframe(df, width="stretch", height=400)


def display_question_detail(results: list[dict], question_idx: int):
    """顯示問題詳情"""
    result = results[question_idx]
    
    # 預處理欄位
    gold_ids = set(result.get("gold_doc_ids", []))
    retrieved_ids = result.get("retrieved_doc_ids", [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**問題:**")
        st.write(result.get("question", ""))
        
        st.markdown("**標準答案:**")
        st.info(result.get("gold_answer", "-"))
        
        st.markdown("**模型回答:**")
        answer = result.get("generated_answer", "-")
        if result.get("is_pass"):
            st.success(answer)
        elif "is_pass" in result:
            st.error(answer)
        else:
            st.write(answer)
        
        if "llm_judgment" in result:
            st.markdown(f"**LLM 判斷:** {result['llm_judgment']}")
    
    with col2:
        # 計算命中
        hit_ids = gold_ids.intersection(retrieved_ids)
        partial_hit = f"{len(hit_ids)}/{len(gold_ids)}"
        
        st.markdown("**統計:**")
        st.write(f"- 命中: {partial_hit}")
        st.write(f"- 回應時間: {result.get('response_time_ms', 0)} ms")
        
        st.markdown("**Gold Doc IDs:**")
        
        # 從資料庫載入 gold doc 內容
        from repositories.document_repository import DocumentRepository
        repo = DocumentRepository()
        
        for doc_id in result.get("gold_doc_ids", []):
            is_hit = doc_id in retrieved_ids
            icon = "✅" if is_hit else "❌"
            
            with st.expander(f"{icon} {doc_id[:20]}..."):
                # 嘗試從檢索結果中找內容
                ctx_content = None
                for ctx in result.get("contexts", result.get("retrieved_contexts", [])):
                    if ctx.get("doc_id") == doc_id:
                        ctx_content = ctx.get("content", ctx.get("content_preview"))
                        break
                
                if ctx_content:
                    st.write(ctx_content)
                else:
                    # 從資料庫查詢
                    doc = repo.find_by_doc_id(doc_id)
                    if doc:
                        st.write(doc.get("content", "無內容"))
                    else:
                        st.write("找不到此文件")
    
    # 檢索到的文件
    contexts = result.get("contexts", result.get("retrieved_contexts", []))
    if contexts:
        st.markdown("### 📚 檢索到的文件")
        for i, ctx in enumerate(contexts):
            doc_id = ctx.get("doc_id", "unknown")
            is_gold = doc_id in gold_ids
            icon = "🎯" if is_gold else "📄"
            score = ctx.get("score", 0)
            
            with st.expander(f"{icon} [{i+1}] {doc_id[:16]}... (Score: {score:.4f})"):
                st.markdown(f"**來源:** {ctx.get('original_source', '-')}")
                content = ctx.get("content", ctx.get("content_preview", "-"))
                st.write(content)


# ===================== 主程式 =====================

def main():
    st.title("🔍 Hybrid RAG 評估儀表板")
    
    # 載入已存在的結果
    if "results" not in st.session_state:
        st.session_state.results = load_existing_results()
    
    # 側邊欄
    with st.sidebar:
        st.header("⚙️ 設定")
        
        mode = st.selectbox("檢索模式", MODES, index=0)
        top_k = st.slider("Top-K", min_value=1, max_value=20, value=5)
        
        st.markdown("---")
        
        if st.button("🚀 執行評估", type="primary", use_container_width=True):  # TODO: width param for button
            with st.spinner("載入查詢..."):
                queries = load_queries()
            
            st.info(f"正在以 {mode} 模式執行 {len(queries)} 題...")
            results = run_evaluation(queries, mode, top_k)
            st.session_state.results[mode] = results
            st.rerun()
        
        if st.button("🔬 執行 LLM 語意評估", use_container_width=True):  # TODO: width param for button
            if mode in st.session_state.results:
                results = run_llm_evaluation(st.session_state.results[mode], mode)
                st.session_state.results[mode] = results
                st.rerun()
            else:
                st.warning("請先執行評估")
        
        if st.button("🔄 重新載入資料", use_container_width=True):  # TODO: width param for button
            st.session_state.results = load_existing_results()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 已有結果")
        for m in MODES:
            if m in st.session_state.results:
                count = len(st.session_state.results[m])
                has_llm = any("is_pass" in r for r in st.session_state.results[m])
                llm_icon = "🔬" if has_llm else ""
                st.write(f"✅ {m} ({count}題) {llm_icon}")
            else:
                st.write(f"⬜ {m}")
    
    # 主區域
    tab1, tab2, tab3 = st.tabs(["📊 指標比較", "📋 結果列表", "🔎 問題詳情"])
    
    with tab1:
        display_metrics_comparison(st.session_state.results)
    
    with tab2:
        if st.session_state.results:
            available_modes = [m for m in MODES if m in st.session_state.results]
            selected_mode = st.selectbox("選擇模式", available_modes, key="results_mode")
            display_results_table(selected_mode, st.session_state.results[selected_mode])
        else:
            st.info("請先執行評估或確認 data 目錄中有結果檔案。")
    
    with tab3:
        if st.session_state.results:
            available_modes = [m for m in MODES if m in st.session_state.results]
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                selected_mode = st.selectbox("模式", available_modes, key="detail_mode")
                results = st.session_state.results[selected_mode]
                
                question_options = [
                    f"{i+1}. {r['question'][:25]}..."
                    for i, r in enumerate(results)
                ]
                selected_q = st.selectbox("選擇問題", question_options, key="detail_q")
                question_idx = question_options.index(selected_q)
            
            with col2:
                display_question_detail(results, question_idx)
        else:
            st.info("請先執行評估。")


if __name__ == "__main__":
    main()
