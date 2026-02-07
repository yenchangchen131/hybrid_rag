#!/usr/bin/env python
"""
計算評估指標

從 RAG 結果檔案計算各項指標，包括：
- Hit Rate (僅針對 single gold doc 的問題)
- Partial Hit Rate: 對於多個 gold_doc_ids 的問題，計算 命中數/總數
- MRR (Mean Reciprocal Rank): 排序精準度指標
- 按問題類型分組統計 (single-hop vs multi-hop)
- 按資料來源分組統計

使用方式：
    uv run python scripts/calculate_metrics.py
    uv run python scripts/calculate_metrics.py --input data/rag_results.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def calculate_reciprocal_rank(gold_ids: set, retrieved_ids: list) -> float:
    """計算 Reciprocal Rank（平均）
    
    對於多個 gold docs，計算每個 gold doc 的 RR 後取平均。
    如果某個 gold doc 沒有被檢索到，該 doc 的 RR = 0
    """
    if not gold_ids:
        return 0.0
    
    rr_sum = 0.0
    for gold_id in gold_ids:
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id == gold_id:
                rr_sum += 1.0 / rank
                break
        # 如果沒找到，RR = 0，不需額外處理
    
    return rr_sum / len(gold_ids)


def calculate_metrics(results: list[dict]) -> dict:
    """計算評估指標"""
    
    # 分開統計：單一 gold doc vs 多個 gold docs
    single_gold_results = []  # 只有一個 gold doc
    multi_gold_results = []   # 多個 gold docs
    
    # 整體統計
    total_gold_docs = 0
    total_hit_docs = 0
    total_rr = 0.0  # 用於計算 MRR
    
    # 按問題類型分組
    by_question_type = defaultdict(lambda: {
        "total": 0, "gold_docs": 0, "hit_docs": 0, "rr_sum": 0.0,
        "single_gold_hits": 0, "single_gold_total": 0
    })
    
    # 按資料來源分組
    by_source = defaultdict(lambda: {
        "total": 0, "gold_docs": 0, "hit_docs": 0, "rr_sum": 0.0,
        "single_gold_hits": 0, "single_gold_total": 0
    })
    
    # 詳細結果
    detailed_results = []
    
    for r in results:
        gold_ids = set(r["gold_doc_ids"])
        retrieved_ids = r["retrieved_doc_ids"]
        
        # 計算命中
        hit_ids = gold_ids.intersection(retrieved_ids)
        hit_count = len(hit_ids)
        gold_count = len(gold_ids)
        is_hit = hit_count > 0
        
        # Reciprocal Rank
        rr = calculate_reciprocal_rank(gold_ids, retrieved_ids)
        total_rr += rr
        
        # 部分命中率字串 (例如 "2/5")
        partial_hit_str = f"{hit_count}/{gold_count}"
        partial_hit_rate = hit_count / gold_count if gold_count > 0 else 0
        
        # 累加整體統計
        total_gold_docs += gold_count
        total_hit_docs += hit_count
        
        # 分類：單一 vs 多個 gold doc
        if gold_count == 1:
            single_gold_results.append(r)
        else:
            multi_gold_results.append(r)
        
        # 分組統計
        q_type = r.get("question_type", "unknown")
        by_question_type[q_type]["total"] += 1
        by_question_type[q_type]["gold_docs"] += gold_count
        by_question_type[q_type]["hit_docs"] += hit_count
        by_question_type[q_type]["rr_sum"] += rr
        if gold_count == 1:
            by_question_type[q_type]["single_gold_total"] += 1
            if is_hit:
                by_question_type[q_type]["single_gold_hits"] += 1
        
        source = r.get("source_dataset", "unknown")
        by_source[source]["total"] += 1
        by_source[source]["gold_docs"] += gold_count
        by_source[source]["hit_docs"] += hit_count
        by_source[source]["rr_sum"] += rr
        if gold_count == 1:
            by_source[source]["single_gold_total"] += 1
            if is_hit:
                by_source[source]["single_gold_hits"] += 1
        
        # 詳細結果
        detailed_results.append({
            "question_id": r["question_id"],
            "question": r["question"],
            "question_type": q_type,
            "source_dataset": source,
            "gold_count": gold_count,
            "gold_doc_ids": list(gold_ids),
            "retrieved_doc_ids": retrieved_ids,
            "hit_doc_ids": list(hit_ids),
            "partial_hit": partial_hit_str,
            "partial_hit_rate": round(partial_hit_rate, 4),
            "reciprocal_rank": round(rr, 4),
        })
    
    # 計算單一 gold doc 的 hit rate
    single_gold_hits = sum(1 for r in single_gold_results 
                          if set(r["gold_doc_ids"]).intersection(r["retrieved_doc_ids"]))
    single_gold_total = len(single_gold_results)
    
    # MRR
    total = len(results)
    mrr = total_rr / total if total > 0 else 0
    
    # 計算分組統計的比率
    def calc_group_stats(group: dict) -> dict:
        stats = {
            "total_questions": group["total"],
            "total_gold_docs": group["gold_docs"],
            "total_hit_docs": group["hit_docs"],
            "partial_hit_rate": round(group["hit_docs"] / group["gold_docs"], 4) if group["gold_docs"] > 0 else 0,
            "mrr": round(group["rr_sum"] / group["total"], 4) if group["total"] > 0 else 0,
        }
        # 只有當有單一 gold doc 的問題時才顯示 hit rate
        if group["single_gold_total"] > 0:
            stats["single_gold_questions"] = group["single_gold_total"]
            stats["single_gold_hit_rate"] = round(group["single_gold_hits"] / group["single_gold_total"], 4)
        return stats
    
    return {
        "summary": {
            "total_questions": total,
            # 單一 gold doc 的 hit rate（避免多 gold doc 的誤導）
            "single_gold_questions": single_gold_total,
            "single_gold_hit_rate": round(single_gold_hits / single_gold_total, 4) if single_gold_total > 0 else None,
            # 多 gold doc 的 partial hit rate
            "multi_gold_questions": len(multi_gold_results),
            "total_gold_docs": total_gold_docs,
            "total_hit_docs": total_hit_docs,
            "partial_hit_rate": round(total_hit_docs / total_gold_docs, 4) if total_gold_docs > 0 else 0,
            # 排序指標
            "mrr": round(mrr, 4),
        },
        "by_question_type": {k: calc_group_stats(v) for k, v in by_question_type.items()},
        "by_source": {k: calc_group_stats(v) for k, v in by_source.items()},
        "detailed_results": detailed_results,
    }


def print_metrics(metrics: dict) -> None:
    """輸出指標"""
    summary = metrics["summary"]
    
    # 按資料來源分組
    print("\n" + "=" * 50)
    print("按資料來源分組")
    print("=" * 50)
    for source, stats in metrics["by_source"].items():
        print(f"\n【{source}】")
        print(f"  問題數:           {stats['total_questions']}")
        if "single_gold_hit_rate" in stats:
            print(f"  Hit Rate:         {stats['single_gold_hit_rate']:.2%}")
        print(f"  Partial Hit Rate: {stats['partial_hit_rate']:.2%} ({stats['total_hit_docs']}/{stats['total_gold_docs']})")
        print(f"  MRR:              {stats['mrr']:.4f}")
    
    # 按問題類型分組
    print("\n" + "=" * 50)
    print("按問題類型分組")
    print("=" * 50)
    for q_type, stats in metrics["by_question_type"].items():
        print(f"\n【{q_type}】")
        print(f"  問題數:           {stats['total_questions']}")
        if "single_gold_hit_rate" in stats:
            print(f"  Hit Rate:         {stats['single_gold_hit_rate']:.2%}")
        print(f"  Partial Hit Rate: {stats['partial_hit_rate']:.2%} ({stats['total_hit_docs']}/{stats['total_gold_docs']})")
        print(f"  MRR:              {stats['mrr']:.4f}")
    
    # 總計
    print("\n" + "=" * 50)
    print("總計")
    print("=" * 50)
    print(f"\n  問題數:           {summary['total_questions']}")
    if summary["single_gold_hit_rate"] is not None:
        print(f"  Hit Rate:         {summary['single_gold_hit_rate']:.2%}")
    print(f"  Partial Hit Rate: {summary['partial_hit_rate']:.2%} ({summary['total_hit_docs']}/{summary['total_gold_docs']})")
    print(f"  MRR:              {summary['mrr']:.4f}")
    print("")


def main():
    parser = argparse.ArgumentParser(description="計算 RAG 評估指標")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/rag_results_hybrid.json",
        help="RAG 結果檔案路徑"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="指標輸出檔案路徑 (預設: 根據 input 自動命名)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # 自動根據 input 檔名推斷 output 檔名
    # rag_results_hybrid.json → evaluation_metrics_hybrid.json
    if args.output:
        output_path = Path(args.output)
    else:
        input_name = input_path.stem  # e.g. "rag_results_hybrid"
        mode_suffix = input_name.replace("rag_results_", "")  # e.g. "hybrid"
        output_path = input_path.parent / f"evaluation_metrics_{mode_suffix}.json"
    
    print("=" * 60)
    print("🔬 計算 RAG 評估指標")
    print("=" * 60)
    
    # 載入結果
    print(f"📂 載入結果: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data.get("results", data)  # 相容舊格式
    print(f"   共 {len(results)} 筆結果")
    
    # 計算指標
    metrics = calculate_metrics(results)
    
    # 輸出到終端
    print_metrics(metrics)
    
    # 儲存結果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 詳細指標已儲存至: {output_path}")


if __name__ == "__main__":
    main()
