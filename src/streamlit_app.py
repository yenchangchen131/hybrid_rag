import streamlit as st
import sys
import os
import json

# 將 src 加入 Python Path 以便匯入 modules
sys.path.append(os.path.join(os.path.dirname(__file__)))

from rag import RAGSystem
from calculate_metrics import load_report
from evaluate import evaluate

# 設定頁面配置
st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")

st.sidebar.title("導覽")
app_mode = st.sidebar.radio("選擇模式", ["Chatbot 問答機器人", "Evaluation Dashboard 評估看板"])

# 初始化 RAG System (使用 cache 避免重複載入)
@st.cache_resource
def load_rag_system():
    return RAGSystem()

try:
    rag = load_rag_system()
except Exception as e:
    st.error(f"系統初始化失敗: {e}")
    st.stop()

if app_mode == "Chatbot 問答機器人":
    st.title("🤖 Hybrid RAG 問答助手")

    # 初始化 Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []

    # Sidebar: 顯示參考來源 (只在 Chatbot 模式顯示與當前對話相關的)
    with st.sidebar:
        st.header("📚 參考來源 (Context)")
        if st.session_state.last_contexts:
            for i, ctx in enumerate(st.session_state.last_contexts):
                doc = ctx['doc']
                score = ctx['score']
                title = doc['metadata'].get('title', '無標題')
                text = doc['text']
                
                with st.expander(f"[{i+1}] {title} (Score: {score:.4f})"):
                    st.write(text)
        else:
            st.info("尚未進行檢索，暫無參考資料。")

    # 顯示對話紀錄
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 處理使用者輸入
    if prompt := st.chat_input("請輸入您的問題..."):
        # 顯示使用者訊息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回答
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # 呼叫 RAG
                answer, contexts = rag.answer(prompt)
                
                # 更新顯示
                message_placeholder.markdown(answer)
                
                # 儲存紀錄
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.last_contexts = contexts
                
                # 強制重新執行以更新 Sidebar
                st.rerun()
                
            except Exception as e:
                message_placeholder.error(f"發生錯誤: {e}")

else: # Evaluation Dashboard 模式
    st.title("📊 Evaluation Dashboard 評估看板")
    
    # 重跑按鈕
    if st.button("🔄 重新執行評測 (這會花一點時間)"):
        progress_bar = st.progress(0, text="準備開始評估...")
        
        def update_progress(current, total):
            progress_bar.progress((current + 1) / total, text=f"正在評估第 {current+1}/{total} 題...")
            
        try:
            evaluate(progress_callback=update_progress)
            st.success("評估完成！請等待畫面重整...")
            st.rerun()
        except Exception as e:
            st.error(f"評估失敗: {e}")
    
    report_path = os.path.join("data", "evaluation_report.json")
    if not os.path.exists(report_path):
        st.warning("找不到評估報告，請先執行 evaluation。")
    else:
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
            
        total = len(report)
        
        # 計算 Metrics
        total_recall_1 = 0
        total_recall_5 = 0
        total_mrr = 0
        
        for item in report:
            gold_id = item['gold_context_id']
            retrieved = item['retrieved_context_ids']
            
            # Recall@1
            if gold_id in retrieved[:1]:
                total_recall_1 += 1
            
            # Recall@5 (原始報告中的 is_hit 也是基於 Top-5)
            if gold_id in retrieved[:5]:
                total_recall_5 += 1
                
            # MRR
            if gold_id in retrieved:
                rank = retrieved.index(gold_id) + 1
                total_mrr += 1.0 / rank
        
        avg_recall_1 = total_recall_1 / total
        avg_recall_5 = total_recall_5 / total
        avg_mrr = total_mrr / total
        
        # 顯示 Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Recall@1", f"{avg_recall_1:.2%}")
        col2.metric("Recall@5", f"{avg_recall_5:.2%}")
        col3.metric("MRR@5", f"{avg_mrr:.4f}")
        
        st.divider()
        st.subheader(f"詳細評測結果 ({total} 筆)")
        
        for i, item in enumerate(report):
            qid = item.get('qid', 'N/A')
            question = item['question']
            gold_answer = item['gold_answer']
            gen_answer = item['generated_answer']
            gold_id = item['gold_context_id']
            retrieved_ids = item['retrieved_context_ids']
            is_hit = item['is_hit']
            
            status_icon = "✅" if is_hit else "❌"
            
            with st.expander(f"{status_icon} [{i+1}] {question}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Ground Truth Answer:**")
                    st.info(gold_answer)
                with c2:
                    st.markdown("**Generated Answer:**")
                    st.success(gen_answer)
                
                st.markdown("---")
                st.markdown(f"**Gold Context ID:** `{gold_id}`")
                
                # 建立 Doc Lookup Map
                if 'doc_map' not in st.session_state:
                    st.session_state.doc_map = {d['doc_id']: d for d in rag.retriever.docs}
                doc_map = st.session_state.doc_map
                
                # 顯示 Gold Content
                gold_text = doc_map.get(gold_id, {}).get('text', 'Content not found')
                with st.expander(f"📖 查看正確答案段落內容 ({gold_id})"):
                    st.info(gold_text)

                st.markdown("**Retrieved Contexts (Top-5):**")
                
                for rank, rid in enumerate(retrieved_ids):
                    is_correct_ctx = (rid == gold_id)
                    rank_display = f"Rank {rank+1}"
                    
                    # 取得內容
                    doc_content = doc_map.get(rid, {}).get('text', 'Content not found')
                    
                    if is_correct_ctx:
                        st.markdown(f"### ✅ {rank_display}: `{rid}` (Correct!)")
                        st.success(doc_content)
                    else:
                        st.markdown(f"### {rank_display}: `{rid}`")
                        st.text(doc_content)
                    
                    st.divider()
                
                if not is_hit:
                    st.error("檢索失敗：正確段落未出現在前 5 筆結果中。")
