#!/usr/bin/env python
"""
批次執行所有查詢並儲存結果

使用方式：
    uv run python scripts/run_all_queries.py
    uv run python scripts/run_all_queries.py --mode vector
    uv run python scripts/run_all_queries.py --mode keyword
    uv run python scripts/run_all_queries.py --mode hybrid
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

from services import RAGService
from core.config import settings
from models.document import QueryModel


VALID_MODES = ["vector", "keyword", "hybrid"]


def load_queries(file_path: Path) -> list[QueryModel]:
    """載入測試查詢"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [QueryModel(**q) for q in data]


def run_all_queries(
    queries: list[QueryModel], 
    rag: RAGService, 
    top_k: int = 5,
    mode: str = "hybrid",
) -> list[dict]:
    """執行所有查詢"""
    results = []
    total_time = 0.0
    
    for query in tqdm(queries, desc=f"執行查詢 ({mode})"):
        # 計時開始
        start_time = time.perf_counter()
        
        response = rag.answer(query.question, top_k=top_k, mode=mode)
        
        # 計時結束
        elapsed_time = time.perf_counter() - start_time
        total_time += elapsed_time
        
        result = {
            "question_id": query.question_id,
            "question": query.question,
            "question_type": query.question_type,
            "source_dataset": query.source_dataset,
            "gold_answer": query.gold_answer,
            "gold_doc_ids": query.gold_doc_ids,
            "generated_answer": response.answer,
            "retrieved_doc_ids": response.retrieved_doc_ids,
            "retrieved_contexts": [
                {
                    "doc_id": ctx.doc_id,
                    "score": ctx.score,
                    "original_source": ctx.original_source,
                    "content_preview": ctx.content[:200] + "..." if len(ctx.content) > 200 else ctx.content,
                }
                for ctx in response.contexts
            ],
            "response_time_ms": round(elapsed_time * 1000, 2),
        }
        results.append(result)
    
    avg_time = (total_time / len(queries)) * 1000 if queries else 0
    print(f"\n⏱️  平均回應時間: {avg_time:.2f} ms")
    
    return results, total_time


def main():
    parser = argparse.ArgumentParser(description="批次執行所有查詢")
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help=f"queries.json 路徑 (預設: {settings.queries_path})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="輸出結果檔案路徑 (預設: data/rag_results_{mode}.json)"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="hybrid",
        choices=VALID_MODES,
        help="檢索模式: vector, keyword, hybrid (預設: hybrid)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="檢索數量 (預設: 5)"
    )
    
    args = parser.parse_args()
    
    queries_path = Path(args.queries) if args.queries else settings.queries_path
    
    # 根據 mode 自動命名輸出檔案
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"data/rag_results_{args.mode}.json")
    
    print("=" * 50)
    print("🔄 批次執行 RAG 查詢")
    print("=" * 50)
    print(f"📌 檢索模式: {args.mode}")
    
    # 載入查詢
    print(f"📂 載入查詢: {queries_path}")
    queries = load_queries(queries_path)
    print(f"   共 {len(queries)} 筆問題")
    
    # 初始化 RAG（根據模式決定是否載入向量索引）
    rag = RAGService()
    rag.initialize(mode=args.mode)
    
    # 執行查詢
    results, total_time = run_all_queries(queries, rag, top_k=args.top_k, mode=args.mode)
    
    # 儲存結果
    output_data = {
        "metadata": {
            "queries_file": str(queries_path),
            "total_questions": len(results),
            "top_k": args.top_k,
            "retrieval_mode": args.mode,
            "total_time_seconds": round(total_time, 2),
            "avg_response_time_ms": round((total_time / len(results)) * 1000, 2) if results else 0,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("=" * 50)
    print(f"✅ 完成！共處理 {len(results)} 筆問題")
    print(f"💾 結果已儲存至: {output_path}")
    print("=" * 50)
    print("\n執行以下指令計算評估指標：")
    print(f"  uv run python scripts/calculate_metrics.py --input {output_path}")


if __name__ == "__main__":
    main()
