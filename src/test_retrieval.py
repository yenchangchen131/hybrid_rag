from retriever import HybridRetriever
import sys

def main():
    print("初始化檢索器...")
    retriever = HybridRetriever()
    
    while True:
        query = input("\n請輸入問題 (輸入 q 退出): ")
        if query.lower() in ['q', 'exit', 'quit']:
            break
            
        print(f"正在搜尋: {query} ...")
        results = retriever.search(query, top_k=5)
        
        if not results:
            print("找不到相關文章。")
        else:
            for i, res in enumerate(results):
                doc = res['doc']
                score = res['score']
                title = doc['metadata']['title']
                text = doc['text']
                # 簡單截斷顯示
                display_text = text[:100].replace("\n", " ") + "..."
                print(f"{i+1}. [{score:.4f}] [{title}] {display_text}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 單次查詢模式 (方便自動化測試)
        query = sys.argv[1]
        retriever = HybridRetriever()
        results = retriever.search(query, top_k=3)
        for i, res in enumerate(results):
            print(f"Rank {i+1}: {res['doc']['metadata']['title']} (Score: {res['score']:.4f})")
    else:
        main()
