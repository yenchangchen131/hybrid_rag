import sys
import os

# 將 src 加入 Python Path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag import RAGSystem

def main():
    print("=== Hybrid RAG 問答系統 ===")
    rag = RAGSystem()
    print("\n系統就緒！請輸入您的問題 (輸入 q, exit, quit 離開)")
    
    while True:
        try:
            query = input("\n[User] > ")
            if query.strip().lower() in ['q', 'exit', 'quit']:
                print("再見！")
                break
            
            if not query.strip():
                continue
                
            answer, contexts = rag.answer(query)
            
            print(f"\n[AI] > {answer}")
            print("-" * 50)
            print("參考來源:")
            for i, ctx in enumerate(contexts):
                title = ctx['doc']['metadata']['title']
                score = ctx['score']
                print(f"{i+1}. {title} (Score: {score:.4f})")
                
        except KeyboardInterrupt:
            print("\n再見！")
            break
        except Exception as e:
            print(f"發生錯誤: {e}")

if __name__ == "__main__":
    main()
