#!/usr/bin/env python
"""
單一查詢測試腳本

使用方式：
    uv run python scripts/run_query.py "台灣於何年開始實施九年國民義務教育?"
    uv run python scripts/run_query.py --query "問題" --top-k 3
"""

import argparse
import sys

from services import RAGService


def main():
    parser = argparse.ArgumentParser(description="測試 RAG 查詢")
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="查詢問題"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        dest="query_arg",
        help="查詢問題（替代位置參數）"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="檢索數量 (預設: 5)"
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="僅執行檢索，不生成答案"
    )
    
    args = parser.parse_args()
    
    # 取得查詢
    query = args.query or args.query_arg
    if not query:
        parser.print_help()
        sys.exit(1)
    
    print("=" * 50)
    print("🔍 Hybrid RAG 查詢")
    print("=" * 50)
    print(f"問題: {query}")
    print("-" * 50)
    
    rag = RAGService()
    
    if args.retrieve_only:
        # 僅檢索
        results = rag.retrieve(query, top_k=args.top_k)
        
        print(f"\n📚 檢索結果 (Top {len(results)}):\n")
        for i, r in enumerate(results):
            print(f"[{i+1}] Score: {r.score:.4f} | Source: {r.original_source}")
            print(f"    {r.content[:100]}...")
            print()
    else:
        # 完整 RAG
        response = rag.answer(query, top_k=args.top_k)
        
        print(f"\n💡 答案:\n{response.answer}")
        print("-" * 50)
        print(f"\n📚 參考來源 ({len(response.contexts)} 筆):")
        for i, ctx in enumerate(response.contexts):
            print(f"  [{i+1}] {ctx.original_source} (Score: {ctx.score:.4f})")


if __name__ == "__main__":
    main()
