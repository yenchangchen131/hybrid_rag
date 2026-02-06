#!/usr/bin/env python
"""
批次評估腳本

使用方式：
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --queries data/queries.json
    uv run python scripts/evaluate.py --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

from services import RAGService
from core.config import settings
from models.document import QueryModel, EvaluationResult


def load_queries(file_path: Path) -> list[QueryModel]:
    """載入測試查詢"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [QueryModel(**q) for q in data]


def evaluate(queries: list[QueryModel], rag: RAGService, top_k: int = 5) -> list[EvaluationResult]:
    """執行評估"""
    results = []
    
    for query in tqdm(queries, desc="評估中"):
        response = rag.answer(query.question, top_k=top_k)
        
        # 計算 Hit Rate
        retrieved_ids = response.retrieved_doc_ids
        gold_ids = set(query.gold_doc_ids)
        hit_count = len(gold_ids.intersection(retrieved_ids))
        is_hit = hit_count > 0
        
        result = EvaluationResult(
            question_id=query.question_id,
            question=query.question,
            gold_answer=query.gold_answer,
            generated_answer=response.answer,
            gold_doc_ids=query.gold_doc_ids,
            retrieved_doc_ids=retrieved_ids,
            is_hit=is_hit,
            hit_count=hit_count,
        )
        results.append(result)
    
    return results


def print_statistics(results: list[EvaluationResult]) -> dict:
    """計算並輸出統計"""
    total = len(results)
    hits = sum(1 for r in results if r.is_hit)
    hit_rate = hits / total if total > 0 else 0
    
    # 按問題類型分組（需要從原始資料取得）
    single_hop = [r for r in results if len(r.gold_doc_ids) == 1]
    multi_hop = [r for r in results if len(r.gold_doc_ids) > 1]
    
    single_hits = sum(1 for r in single_hop if r.is_hit)
    multi_hits = sum(1 for r in multi_hop if r.is_hit)
    
    stats = {
        "total_questions": total,
        "total_hits": hits,
        "hit_rate": hit_rate,
        "single_hop": {
            "total": len(single_hop),
            "hits": single_hits,
            "hit_rate": single_hits / len(single_hop) if single_hop else 0,
        },
        "multi_hop": {
            "total": len(multi_hop),
            "hits": multi_hits,
            "hit_rate": multi_hits / len(multi_hop) if multi_hop else 0,
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    print("\n" + "=" * 50)
    print("📊 評估結果統計")
    print("=" * 50)
    print(f"總題數:       {total}")
    print(f"命中數:       {hits}")
    print(f"Hit Rate:     {hit_rate:.2%}")
    print("-" * 50)
    print(f"Single-hop:   {single_hits}/{len(single_hop)} ({stats['single_hop']['hit_rate']:.2%})")
    print(f"Multi-hop:    {multi_hits}/{len(multi_hop)} ({stats['multi_hop']['hit_rate']:.2%})")
    print("=" * 50)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="批次評估 RAG 系統")
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help=f"queries.json 路徑 (預設: {settings.queries_path})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/evaluation_results.json",
        help="輸出結果檔案路徑"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="檢索數量 (預設: 5)"
    )
    
    args = parser.parse_args()
    
    queries_path = Path(args.queries) if args.queries else settings.queries_path
    output_path = Path(args.output)
    
    print("=" * 50)
    print("🔬 Hybrid RAG 批次評估")
    print("=" * 50)
    
    # 載入查詢
    print(f"📂 載入查詢: {queries_path}")
    queries = load_queries(queries_path)
    print(f"   共 {len(queries)} 筆測試題")
    
    # 初始化 RAG
    rag = RAGService()
    rag.initialize()
    
    # 執行評估
    results = evaluate(queries, rag, top_k=args.top_k)
    
    # 輸出統計
    stats = print_statistics(results)
    
    # 儲存結果
    output_data = {
        "statistics": stats,
        "results": [r.model_dump() for r in results],
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 詳細結果已儲存至: {output_path}")


if __name__ == "__main__":
    main()
