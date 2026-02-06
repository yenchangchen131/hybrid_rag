"""
文件與查詢資料模型

定義 corpus.json 與 queries.json 的資料結構。
"""

from typing import Literal
from pydantic import BaseModel, Field


class DocumentModel(BaseModel):
    """文件模型 - 對應 corpus.json 結構"""
    
    doc_id: str = Field(..., description="文件唯一識別符 (UUID)")
    content: str = Field(..., description="文件內容")
    original_source: str = Field(..., description="來源資料集 (drcd/squad/hotpotqa/2wiki)")
    original_id: str = Field(..., description="原始資料集中的ID")
    is_gold: bool = Field(default=False, description="是否為黃金標準文件")


class DocumentInDB(DocumentModel):
    """資料庫中的文件模型 - 包含 embedding"""
    
    embedding: list[float] | None = Field(default=None, description="向量嵌入")


class QueryModel(BaseModel):
    """查詢模型 - 對應 queries.json 結構"""
    
    question_id: str = Field(..., description="問題唯一識別符")
    question: str = Field(..., description="問題文字")
    gold_answer: str = Field(..., description="標準答案")
    gold_doc_ids: list[str] = Field(..., description="相關文件ID列表")
    source_dataset: str = Field(..., description="來源資料集")
    question_type: Literal["single-hop", "multi-hop"] = Field(..., description="問題類型")


class EvaluationResult(BaseModel):
    """單一問題的評估結果"""
    
    question_id: str
    question: str
    gold_answer: str
    generated_answer: str
    gold_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    is_hit: bool = Field(..., description="是否命中黃金文件")
    hit_count: int = Field(default=0, description="命中的黃金文件數量")
