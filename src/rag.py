from retriever import HybridRetriever
from generator import Generator

class RAGSystem:
    def __init__(self):
        print("初始化 RAG 系統 (載入檢索器與生成模型)...")
        self.retriever = HybridRetriever()
        self.generator = Generator()

    def answer(self, query):
        print(f"\n[RAG] 正在檢索: {query}")
        
        # 1. 檢索
        retrieved_results = self.retriever.search(query, top_k=5)
        
        if not retrieved_results:
            return "抱歉，找不到相關資訊。", []
            
        print(f"[RAG] 找到 {len(retrieved_results)} 筆相關資料，正在生成回答...")
        
        # 2. 生成
        answer = self.generator.generate_answer(query, retrieved_results)
        
        return answer, retrieved_results
