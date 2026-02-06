"""
RAG 整合服務

整合檢索與生成服務，提供完整的 RAG 流程。
"""

from core.config import settings
from models.response import RAGResponse, RetrievalResult
from services.retrieval_service import RetrievalService
from services.generation_service import GenerationService


class RAGService:
    """RAG 整合服務"""
    
    def __init__(self):
        self._retrieval_service = RetrievalService()
        self._generation_service = GenerationService()
        self._vector_index_loaded = False
    
    def initialize(self, mode: str = "hybrid") -> None:
        """初始化服務
        
        Args:
            mode: 檢索模式，只有 vector/hybrid 需要載入向量索引
        """
        print("🚀 正在初始化 RAG 系統...")
        
        # 預先建立 MongoDB 連線（避免後續與 tqdm 輸出交錯）
        from repositories.document_repository import DocumentRepository
        _ = DocumentRepository().count()
        
        # 只有需要向量檢索時才載入向量索引
        if mode in ("vector", "hybrid") and not self._vector_index_loaded:
            self._retrieval_service.load_vector_index()
            self._vector_index_loaded = True
        elif mode == "keyword":
            print("📌 Keyword 模式，跳過向量索引載入")
        
        print("✅ RAG 系統就緒")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int | None = None,
        mode: str = "hybrid",
    ) -> list[RetrievalResult]:
        """僅執行檢索
        
        Args:
            query: 查詢文字
            top_k: 返回數量
            mode: 檢索模式 ("vector", "keyword", "hybrid")
            
        Returns:
            檢索結果
        """
        # 確保需要向量時已載入
        if mode in ("vector", "hybrid") and not self._vector_index_loaded:
            self._retrieval_service.load_vector_index()
            self._vector_index_loaded = True
        
        return self._retrieval_service.search(query, top_k, mode=mode)
    
    def answer(
        self, 
        query: str, 
        top_k: int | None = None,
        mode: str = "hybrid",
    ) -> RAGResponse:
        """完整 RAG 流程：檢索 + 生成
        
        Args:
            query: 使用者問題
            top_k: 檢索數量
            mode: 檢索模式 ("vector", "keyword", "hybrid")
            
        Returns:
            RAG 回應（包含答案與上下文）
        """
        # 確保需要向量時已載入
        if mode in ("vector", "hybrid") and not self._vector_index_loaded:
            self._retrieval_service.load_vector_index()
            self._vector_index_loaded = True
        
        # 檢索
        contexts = self._retrieval_service.search(query, top_k, mode=mode)
        
        if not contexts:
            return RAGResponse(
                query=query,
                answer="抱歉，找不到相關資訊。",
                contexts=[],
            )
        
        # 生成
        answer = self._generation_service.generate_answer(query, contexts)
        
        return RAGResponse(
            query=query,
            answer=answer,
            contexts=contexts,
        )
