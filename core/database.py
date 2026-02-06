"""
MongoDB 資料庫連線管理

使用單例模式確保整個應用程式共用同一個連線。
"""

from typing import ClassVar
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from core.config import settings


class MongoDBClient:
    """MongoDB 連線管理器（單例模式）"""
    
    _instance: ClassVar["MongoDBClient | None"] = None
    _client: MongoClient | None = None
    _db: Database | None = None
    
    def __new__(cls) -> "MongoDBClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def connect(self) -> None:
        """建立 MongoDB 連線"""
        if self._client is not None:
            return
            
        try:
            self._client = MongoClient(settings.mongodb_uri)
            self._db = self._client[settings.MONGODB_DB_NAME]
            # 測試連線
            self._client.admin.command("ping")
            print(f"✅ MongoDB 連線成功: {settings.MONGODB_DB_NAME}")
        except Exception as e:
            print(f"❌ MongoDB 連線失敗: {e}")
            raise
    
    @property
    def db(self) -> Database:
        """取得資料庫實例"""
        if self._db is None:
            self.connect()
        return self._db  # type: ignore
    
    def get_collection(self, name: str | None = None) -> Collection:
        """取得指定 Collection
        
        Args:
            name: Collection 名稱，預設使用配置中的 COLLECTION_NAME
        """
        collection_name = name or settings.COLLECTION_NAME
        return self.db[collection_name]
    
    def close(self) -> None:
        """關閉連線"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            print("MongoDB 連線已關閉")
    
    @classmethod
    def get_instance(cls) -> "MongoDBClient":
        """取得單例實例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# 便捷函數
def get_db() -> MongoDBClient:
    """取得 MongoDB 客戶端單例"""
    return MongoDBClient.get_instance()
