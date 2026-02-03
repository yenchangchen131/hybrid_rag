import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from db_manager import MongoDBManager
from embedding import get_embedding

class HybridRetriever:
    def __init__(self):
        self.db_manager = MongoDBManager()
        self.collection = self.db_manager.get_collection("drcd_knowledge_base")
        
        # 為了模擬 Vector Search (在非 Atlas 環境)，我們預先載入所有向量到記憶體
        # 注意：這只適合小規模數據 (如本專案 500 筆)
        print("正在載入向量索引...")
        self.docs = []
        self.embeddings = []
        
        cursor = self.collection.find({"embedding": {"$ne": None}})
        for doc in cursor:
            self.docs.append(doc)
            self.embeddings.append(doc['embedding'])
            
        if self.embeddings:
            self.embeddings = np.array(self.embeddings)
            print(f"向量索引載入完成，共 {len(self.embeddings)} 筆。")
        else:
            print("警告：資料庫中無向量資料！")

    def vector_search(self, query, top_k=10):
        """語意搜尋 (Dense Retrieval)"""
        query_emb = get_embedding(query)
        if not query_emb or len(self.embeddings) == 0:
            return []
        
        # 計算餘弦相似度
        query_emb = np.array(query_emb).reshape(1, -1)
        sim_scores = cosine_similarity(query_emb, self.embeddings)[0]
        
        # 取得 Top-K 索引
        top_indices = sim_scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            doc = self.docs[idx]
            results.append({
                "doc": doc,
                "score": float(sim_scores[idx]),
                "type": "vector"
            })
        return results

    def keyword_search(self, query, top_k=10):
        """關鍵字搜尋 (Sparse Retrieval) - 使用 MongoDB Text Search"""
        # MongoDB 的 Text Search 預設使用 OR 邏輯
        cursor = self.collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(top_k)
        
        results = []
        for doc in cursor:
            results.append({
                "doc": doc,
                "score": doc.get('score', 0),
                "type": "keyword"
            })
        return results

    def rrf_fusion(self, vector_results, keyword_results, k=60):
        """Reciprocal Rank Fusion (RRF)"""
        fused_scores = {}
        doc_map = {}
        
        # 處理 Vector 結果
        for rank, item in enumerate(vector_results):
            doc_id = str(item['doc']['_id'])
            doc_map[doc_id] = item['doc']
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + rank + 1)
            
        # 處理 Keyword 結果
        for rank, item in enumerate(keyword_results):
            doc_id = str(item['doc']['_id'])
            doc_map[doc_id] = item['doc']
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + rank + 1)
        
        # 排序
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        fused_results = []
        for doc_id in sorted_ids:
            fused_results.append({
                "doc": doc_map[doc_id],
                "score": fused_scores[doc_id],
                "type": "hybrid"
            })
        
        return fused_results

    def search(self, query, top_k=5):
        """混合搜尋主入口"""
        # 兩路召回 (為了融合效果，通常各自取多一點，例如 top_k * 2 或固定 20)
        initial_k = top_k * 4 
        
        vec_res = self.vector_search(query, top_k=initial_k)
        key_res = self.keyword_search(query, top_k=initial_k)
        
        # 融合並重排序
        fused_results = self.rrf_fusion(vec_res, key_res)
        
        # 取最終 Top-K
        return fused_results[:top_k]

if __name__ == "__main__":
    # Test
    retriever = HybridRetriever()
    results = retriever.search("颱風是如何形成的？", top_k=3)
    for r in results:
        print(f"[{r['score']:.4f}] {r['doc']['metadata']['title']}: {r['doc']['text'][:50]}...")
