"""
RAG 回應資料模型
"""

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """檢索結果項目"""
    
    doc_id: str
    content: str
    score: float
    retrieval_type: str = Field(..., description="檢索類型: vector/keyword/hybrid")
    original_source: str | None = None


class RAGResponse(BaseModel):
    """RAG 完整回應"""
    
    query: str = Field(..., description="原始查詢")
    answer: str = Field(..., description="生成的答案")
    contexts: list[RetrievalResult] = Field(default_factory=list, description="檢索到的上下文")
    
    @property
    def retrieved_doc_ids(self) -> list[str]:
        """取得所有檢索到的文件ID"""
        return [ctx.doc_id for ctx in self.contexts]
