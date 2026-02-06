"""
資料導入服務

ETL 流程：讀取 corpus.json → 生成 embedding → 寫入 MongoDB
"""

import json
from pathlib import Path
from tqdm import tqdm

from core.config import settings
from repositories.document_repository import DocumentRepository
from services.embedding_service import EmbeddingService


class IngestionService:
    """資料導入服務"""
    
    def __init__(self):
        self._repository = DocumentRepository()
        self._embedding_service = EmbeddingService()
    
    def load_corpus(self, file_path: Path | str) -> list[dict]:
        """讀取 corpus.json
        
        Args:
            file_path: 檔案路徑
            
        Returns:
            文件列表
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"找不到檔案: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            documents = json.load(f)
        
        print(f"📄 已讀取 {len(documents)} 筆文件")
        return documents
    
    def ingest_corpus(
        self,
        file_path: Path | str | None = None,
        generate_embeddings: bool = True,
        batch_size: int = 50,
        clear_existing: bool = True,
    ) -> int:
        """完整的 ETL 導入流程
        
        Args:
            file_path: corpus.json 路徑，預設使用配置值
            generate_embeddings: 是否生成 embedding
            batch_size: 批次處理大小
            clear_existing: 是否清除現有資料
            
        Returns:
            成功導入的文件數量
        """
        path = Path(file_path) if file_path else settings.corpus_path
        
        # 1. 讀取資料
        documents = self.load_corpus(path)
        
        if not documents:
            print("⚠️ 沒有資料可導入")
            return 0
        
        # 2. 清除舊資料
        if clear_existing:
            deleted = self._repository.delete_all()
            print(f"🗑️ 已清除 {deleted} 筆舊資料")
        
        # 3. 生成 embeddings（分批處理）
        if generate_embeddings:
            print("🔄 正在生成 embeddings...")
            contents = [doc["content"] for doc in documents]
            
            all_embeddings: list[list[float]] = []
            for i in tqdm(range(0, len(contents), batch_size), desc="Embedding"):
                batch = contents[i:i + batch_size]
                batch_embeddings = self._embedding_service.get_embeddings_batch(batch)
                all_embeddings.extend(batch_embeddings)
            
            # 將 embedding 加入文件
            for doc, emb in zip(documents, all_embeddings):
                doc["embedding"] = emb if emb else None
        
        # 4. 寫入資料庫
        print("💾 正在寫入 MongoDB...")
        inserted = self._repository.insert_many(documents)
        
        # 5. 建立索引
        self._repository.create_indexes()
        
        print(f"✅ 成功導入 {inserted} 筆文件")
        return inserted
    
    def update_embeddings_only(self, batch_size: int = 50) -> int:
        """只更新現有文件的 embeddings（不重新導入）
        
        Args:
            batch_size: 批次大小
            
        Returns:
            更新的數量
        """
        # 取得所有沒有 embedding 的文件
        collection = self._repository.collection
        cursor = collection.find({"embedding": None})
        docs_without_embedding = list(cursor)
        
        if not docs_without_embedding:
            print("ℹ️ 所有文件都已有 embedding")
            return 0
        
        print(f"🔄 需要生成 {len(docs_without_embedding)} 筆 embedding...")
        
        updated = 0
        for i in tqdm(range(0, len(docs_without_embedding), batch_size), desc="Updating"):
            batch = docs_without_embedding[i:i + batch_size]
            contents = [doc["content"] for doc in batch]
            embeddings = self._embedding_service.get_embeddings_batch(contents)
            
            for doc, emb in zip(batch, embeddings):
                if emb:
                    self._repository.update_embedding(doc["doc_id"], emb)
                    updated += 1
        
        print(f"✅ 已更新 {updated} 筆 embedding")
        return updated
