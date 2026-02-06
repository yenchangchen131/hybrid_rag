#!/usr/bin/env python
"""
Hybrid RAG 問答系統入口

使用方式：
    uv run python main.py
"""

from services import RAGService


def main():
    print("=" * 50)
    print("🤖 Hybrid RAG 問答系統")
    print("=" * 50)
    
    rag = RAGService()
    rag.initialize()
    
    print("\n系統就緒！請輸入您的問題 (輸入 q, exit, quit 離開)")
    
    while True:
        try:
            query = input("\n[User] > ")
            
            if query.strip().lower() in ['q', 'exit', 'quit']:
                print("再見！")
                break
            
            if not query.strip():
                continue
            
            response = rag.answer(query)
            
            print(f"\n[AI] > {response.answer}")
            print("-" * 50)
            print("📚 參考來源:")
            for i, ctx in enumerate(response.contexts):
                source = ctx.original_source or "unknown"
                print(f"  {i+1}. {source} (Score: {ctx.score:.4f})")
                
        except KeyboardInterrupt:
            print("\n再見！")
            break
        except Exception as e:
            print(f"❌ 發生錯誤: {e}")


if __name__ == "__main__":
    main()
