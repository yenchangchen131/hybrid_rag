import os
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.operations import IndexModel

class MongoDBManager:
    def __init__(self, host='localhost', port=27017, db_name='hybrid_rag'):
        self.host = host
        self.port = port
        self.db_name = db_name
        self.client = None
        self.db = None

    def connect(self):
        """建立 MongoDB 連線"""
        try:
            self.client = MongoClient(f"mongodb://{self.host}:{self.port}/")
            self.db = self.client[self.db_name]
            # 測試連線
            self.client.admin.command('ping')
            print("MongoDB 連線成功")
        except Exception as e:
            print(f"MongoDB 連線失敗: {e}")
            raise

    def get_collection(self, collection_name):
        """取得指定 Collection"""
        if self.db is None:
            self.connect()
        return self.db[collection_name]

    def insert_documents(self, collection_name, documents):
        """插入多筆文件"""
        collection = self.get_collection(collection_name)
        try:
            result = collection.insert_many(documents)
            print(f"成功插入 {len(result.inserted_ids)} 筆資料到 {collection_name}")
            return result
        except Exception as e:
            print(f"插入資料失敗: {e}")
            raise

    def create_vector_index(self, collection_name, embedding_field="embedding"):
        """
        建立向量搜尋索引 (Local Mongo 暫時使用標準索引或僅儲存)
        注意：MongoDB Atlas Search 的 Vector Search 需要在 Cloud Console 設定，
        本地端標準 MongoDB 不支援 Atlas Vector Search 語法。
        這裡我們僅建立一般索引以利未來擴充或查詢。
        """
        collection = self.get_collection(collection_name)
        # 這裡僅建立基本的 metadata 索引，實際向量搜尋在本地端可能需要依賴
        # 1. 載入所有向量到記憶體用 numpy/faiss 計算
        # 2. 或者僅作為儲存，檢索時取出。
        
        # 建立 metadata 索引
        collection.create_index([("doc_id", ASCENDING)], unique=True)
        collection.create_index([("metadata.title", ASCENDING)])
        print(f"已建立 {collection_name} 的基礎索引")

    def create_text_index(self, collection_name, fields):
        """建立全文檢索索引 (Text Index)"""
        collection = self.get_collection(collection_name)
        index_fields = [(field, "text") for field in fields]
        collection.create_index(index_fields, name="text_search_index")
        print(f"已建立 {collection_name} 的 Text Index: {fields}")

if __name__ == "__main__":
    # 測試連線
    db = MongoDBManager()
    try:
        db.connect()
    except:
        pass
