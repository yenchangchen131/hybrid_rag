import json
import os

def load_report(filename="data/evaluation_report.json"):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_metrics(report):
    total = len(report)
    if total == 0:
        print("No data to evaluate.")
        return

    # 要計算的 K 值
    k_list = [1, 2, 5]
    
    for k in k_list:
        total_recall = 0
        total_precision = 0
        total_mrr = 0
        
        for item in report:
            gold_id = item['gold_context_id']
            # 取前 K 個結果
            retrieved_ids = item['retrieved_context_ids'][:k]
            
            # 1. Recall@K (是否在 Top-K 中找到)
            is_hit = gold_id in retrieved_ids
            recall = 1.0 if is_hit else 0.0
            
            # 2. Precision@K (Top-K 中正確的比例)
            # 因為只有 1 個正確答案，如果 Hit，Precision = 1/K，否則 0
            precision = 1.0 / k if is_hit else 0.0
            
            # 3. MRR (Mean Reciprocal Rank)
            mrr = 0.0
            if is_hit:
                # 找出 rank (1-based)
                rank = retrieved_ids.index(gold_id) + 1
                mrr = 1.0 / rank
                
            total_recall += recall
            total_precision += precision
            total_mrr += mrr
            
        avg_recall = total_recall / total
        avg_precision = total_precision / total
        avg_mrr = total_mrr / total
        
        print(f"=== Evaluation Metrics (Top-{k}) ===")
        print(f"Total Queries: {total}")
        print(f"Recall@{k}    : {avg_recall:.4f}")
        print(f"Precision@{k} : {avg_precision:.4f}")
        print(f"MRR@{k}       : {avg_mrr:.4f}")
        print("================================\n")

if __name__ == "__main__":
    report = load_report()
    calculate_metrics(report)
