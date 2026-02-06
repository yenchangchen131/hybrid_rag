#!/usr/bin/env python
"""
資料導入腳本

使用方式：
    uv run python scripts/ingest_data.py
    uv run python scripts/ingest_data.py --corpus data/corpus.json
    uv run python scripts/ingest_data.py --no-embeddings
"""

import argparse
import sys

from services import IngestionService
from core.config import settings


def main():
    parser = argparse.ArgumentParser(description="導入 corpus.json 到 MongoDB")
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help=f"corpus.json 路徑 (預設: {settings.corpus_path})"
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="不生成 embeddings（資料導入後需另外執行）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Embedding 批次大小 (預設: 50)"
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="保留現有資料（不清除）"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🚀 Hybrid RAG 資料導入工具")
    print("=" * 50)
    
    service = IngestionService()
    
    try:
        count = service.ingest_corpus(
            file_path=args.corpus,
            generate_embeddings=not args.no_embeddings,
            batch_size=args.batch_size,
            clear_existing=not args.keep_existing,
        )
        
        print("=" * 50)
        print(f"✅ 導入完成！共 {count} 筆文件")
        
    except FileNotFoundError as e:
        print(f"❌ 錯誤: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 導入失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
