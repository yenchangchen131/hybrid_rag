"""
文件資料庫操作層

封裝所有與 MongoDB 的 CRUD 操作。
"""

from typing import Any
from pymongo import ASCENDING
from pymongo.collection import Collection

from core.database import get_db
from core.config import settings
from models.document import DocumentModel, DocumentInDB


class DocumentRepository:
    """文件資料庫操作類"""
    
    def __init__(self, collection_name: str | None = None):
        """初始化 Repository
        
        Args:
            collection_name: Collection 名稱，預設使用配置值
        """
        self._db = get_db()
        self._collection_name = collection_name or settings.COLLECTION_NAME
    
    @property
    def collection(self) -> Collection:
        """取得 Collection"""
        return self._db.get_collection(self._collection_name)
    
    def insert_many(self, documents: list[dict[str, Any]]) -> int:
        """批次插入文件
        
        Args:
            documents: 文件列表（字典格式）
            
        Returns:
            成功插入的數量
        """
        from pymongo.errors import BulkWriteError
        
        if not documents:
            return 0
        
        try:
            # ordered=False: 即使有重複也繼續插入其他文件
            result = self.collection.insert_many(documents, ordered=False)
            return len(result.inserted_ids)
        except BulkWriteError as e:
            # 回傳實際成功插入的數量
            inserted = e.details.get("nInserted", 0)
            errors = len(e.details.get("writeErrors", []))
            print(f"⚠️ 插入時有 {errors} 筆錯誤（重複或其他），成功 {inserted} 筆")
            return inserted
    
    def find_by_doc_id(self, doc_id: str) -> dict[str, Any] | None:
        """根據 doc_id 查詢文件"""
        return self.collection.find_one({"doc_id": doc_id})
    
    def find_by_doc_ids(self, doc_ids: list[str]) -> list[dict[str, Any]]:
        """根據多個 doc_id 查詢文件"""
        cursor = self.collection.find({"doc_id": {"$in": doc_ids}})
        return list(cursor)
    
    def get_all_with_embeddings(self) -> list[dict[str, Any]]:
        """取得所有有 embedding 的文件（用於向量搜尋）"""
        cursor = self.collection.find({"embedding": {"$ne": None}})
        return list(cursor)
    
    def count(self) -> int:
        """取得文件總數"""
        return self.collection.count_documents({})
    
    def count_with_embeddings(self) -> int:
        """取得有 embedding 的文件數量"""
        return self.collection.count_documents({"embedding": {"$ne": None}})
    
    def text_search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """全文檢索
        
        Args:
            query: 搜尋字串
            limit: 最大回傳數量
            
        Returns:
            包含分數的文件列表
        """
        cursor = self.collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        
        return list(cursor)
    
    def delete_all(self) -> int:
        """刪除所有文件
        
        Returns:
            刪除的數量
        """
        result = self.collection.delete_many({})
        return result.deleted_count
    
    def update_embedding(self, doc_id: str, embedding: list[float]) -> bool:
        """更新文件的 embedding
        
        Args:
            doc_id: 文件ID
            embedding: 向量嵌入
            
        Returns:
            是否更新成功
        """
        result = self.collection.update_one(
            {"doc_id": doc_id},
            {"$set": {"embedding": embedding}}
        )
        return result.modified_count > 0
    
    def create_indexes(self) -> None:
        """建立索引"""
        # doc_id 唯一索引
        self.collection.create_index([("doc_id", ASCENDING)], unique=True)
        
        # 來源索引
        self.collection.create_index([("original_source", ASCENDING)])
        
        # 全文檢索索引
        try:
            self.collection.create_index(
                [("content", "text")],
                name="content_text_index",
                default_language="none"  # 支援中文
            )
        except Exception as e:
            # 索引可能已存在
            print(f"Text index 建立備註: {e}")
        
        print(f"✅ 已建立 {self._collection_name} 的索引")
