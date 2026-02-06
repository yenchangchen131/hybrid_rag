"""
檢索服務

實作混合式檢索：Vector Search + Keyword Search + RRF Fusion
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from core.config import settings
from repositories.document_repository import DocumentRepository
from models.response import RetrievalResult
from services.embedding_service import EmbeddingService


class RetrievalService:
    """混合式檢索服務"""
    
    def __init__(self):
        self._repository = DocumentRepository()
        self._embedding_service = EmbeddingService()
        
        # 向量索引（記憶體中）
        self._docs: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._index_loaded = False
    
    def load_vector_index(self) -> None:
        """載入向量索引到記憶體
        
        注意：適用於小規模資料集（< 10K 文件）
        大規模應使用 FAISS 或 MongoDB Atlas Vector Search
        """
        if self._index_loaded:
            return
        
        print("📂 正在載入向量索引...")
        docs_with_embeddings = self._repository.get_all_with_embeddings()
        
        self._docs = []
        embeddings_list = []
        
        for doc in docs_with_embeddings:
            if doc.get("embedding"):
                self._docs.append(doc)
                embeddings_list.append(doc["embedding"])
        
        if embeddings_list:
            self._embeddings = np.array(embeddings_list)
            print(f"✅ 向量索引載入完成: {len(self._docs)} 筆文件")
        else:
            print("⚠️ 沒有可用的向量資料")
            self._embeddings = None
        
        self._index_loaded = True
    
    def vector_search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """語意搜尋（Dense Retrieval）
        
        Args:
            query: 查詢文字
            top_k: 返回數量
            
        Returns:
            檢索結果列表
        """
        if not self._index_loaded:
            self.load_vector_index()
        
        if self._embeddings is None or len(self._embeddings) == 0:
            return []
        
        # 取得查詢向量
        query_embedding = self._embedding_service.get_embedding(query)
        if not query_embedding:
            return []
        
        # 計算餘弦相似度
        query_vec = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_vec, self._embeddings)[0]
        
        # 取 Top-K
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            doc = self._docs[idx]
            results.append(RetrievalResult(
                doc_id=doc["doc_id"],
                content=doc["content"],
                score=float(similarities[idx]),
                retrieval_type="vector",
                original_source=doc.get("original_source"),
            ))
        
        return results
    
    def keyword_search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """關鍵字搜尋（Sparse Retrieval）
        
        Args:
            query: 查詢文字
            top_k: 返回數量
            
        Returns:
            檢索結果列表
        """
        docs = self._repository.text_search(query, limit=top_k)
        
        results = []
        for doc in docs:
            results.append(RetrievalResult(
                doc_id=doc["doc_id"],
                content=doc["content"],
                score=doc.get("score", 0.0),
                retrieval_type="keyword",
                original_source=doc.get("original_source"),
            ))
        
        return results
    
    def rrf_fusion(
        self,
        vector_results: list[RetrievalResult],
        keyword_results: list[RetrievalResult],
        k: int | None = None,
    ) -> list[RetrievalResult]:
        """Reciprocal Rank Fusion (RRF)
        
        Args:
            vector_results: 向量搜尋結果
            keyword_results: 關鍵字搜尋結果
            k: RRF 參數
            
        Returns:
            融合後的結果
        """
        k = k or settings.RRF_K
        
        fused_scores: dict[str, float] = {}
        doc_map: dict[str, RetrievalResult] = {}
        
        # 處理向量結果
        for rank, result in enumerate(vector_results):
            doc_id = result.doc_id
            doc_map[doc_id] = result
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
        # 處理關鍵字結果
        for rank, result in enumerate(keyword_results):
            doc_id = result.doc_id
            if doc_id not in doc_map:
                doc_map[doc_id] = result
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
        # 排序
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        # 建立結果
        fused_results = []
        for doc_id in sorted_ids:
            original = doc_map[doc_id]
            fused_results.append(RetrievalResult(
                doc_id=original.doc_id,
                content=original.content,
                score=fused_scores[doc_id],
                retrieval_type="hybrid",
                original_source=original.original_source,
            ))
        
        return fused_results
    
    def search(
        self, 
        query: str, 
        top_k: int | None = None,
        mode: str = "hybrid",
    ) -> list[RetrievalResult]:
        """檢索主入口
        
        Args:
            query: 查詢文字
            top_k: 最終返回數量
            mode: 檢索模式 ("vector", "keyword", "hybrid")
            
        Returns:
            檢索結果列表
        """
        top_k = top_k or settings.DEFAULT_TOP_K
        initial_k = settings.INITIAL_RETRIEVAL_K
        
        if mode == "vector":
            # 純向量檢索
            return self.vector_search(query, top_k=top_k)
        
        elif mode == "keyword":
            # 純關鍵字檢索
            return self.keyword_search(query, top_k=top_k)
        
        else:
            # 混合檢索 (hybrid)
            vector_results = self.vector_search(query, top_k=initial_k)
            keyword_results = self.keyword_search(query, top_k=initial_k)
            fused_results = self.rrf_fusion(vector_results, keyword_results)
            return fused_results[:top_k]

