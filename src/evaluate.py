import json
import os
import sys
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag import RAGSystem

def load_validation_set(filename="data/drcd_validation_set.json"):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate(progress_callback=None):
    val_data = load_validation_set()
    rag = RAGSystem()
    
    print(f"開始評估，共有 {len(val_data)} 筆測試資料...")
    
    results = []
    retrieval_hits = 0
    
    total = len(val_data)
    for index, item in enumerate(val_data):
        if progress_callback:
            progress_callback(index, total)
        qid = item['qid']
        question = item['question']
        gold_answer = item['ground_truth_answer']
        gold_context_id = item['context_id']
        
        # 執行 RAG
        # 為了評測 Retrieve Hit Rate，我們需要檢索結果
        # RAGSystem.answer 回傳 (answer, contexts)
        generated_answer, retrieved_contexts = rag.answer(question)
        
        # 計算 Hit Rate
        # 檢查 retrieved_contexts 中是否有 doc_id == gold_context_id
        is_hit = False
        retrieved_ids = []
        for ctx in retrieved_contexts:
            doc_id = ctx['doc']['doc_id']
            retrieved_ids.append(doc_id)
            if doc_id == gold_context_id:
                is_hit = True
                
        if is_hit:
            retrieval_hits += 1
            
        results.append({
            "qid": qid,
            "question": question,
            "gold_answer": gold_answer,
            "generated_answer": generated_answer,
            "gold_context_id": gold_context_id,
            "retrieved_context_ids": retrieved_ids,
            "is_hit": is_hit
        })

    # 統計
    hit_rate = retrieval_hits / len(val_data)
    print("\n" + "="*30)
    print(f"評估完成！")
    print(f"Total Questions: {len(val_data)}")
    print(f"Retrieval Hit Rate: {hit_rate:.2%} ({retrieval_hits}/{len(val_data)})")
    print("="*30)
    
    # 儲存詳細報告
    output_file = "data/evaluation_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"詳細評估報告已儲存至 {output_file}")

    # 顯示前幾筆範例
    print("\n--- 範例輸出 ---")
    for res in results[:3]:
        print(f"Q: {res['question']}")
        print(f"Gold A: {res['gold_answer']}")
        print(f"Gen A:  {res['generated_answer']}")
        print(f"Hit: {res['is_hit']}")
        print("-" * 20)

if __name__ == "__main__":
    evaluate()
