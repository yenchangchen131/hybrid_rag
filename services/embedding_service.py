"""
Embedding 服務

封裝 OpenAI Embedding API 調用。
"""

from openai import OpenAI

from core.config import settings


class EmbeddingService:
    """向量嵌入服務"""
    
    def __init__(self):
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.EMBEDDING_MODEL
    
    def get_embedding(self, text: str) -> list[float]:
        """取得單一文字的 embedding
        
        Args:
            text: 輸入文字
            
        Returns:
            向量嵌入（1536 維 for text-embedding-3-small）
        """
        # 清理文字
        text = text.replace("\n", " ").strip()
        if not text:
            return []
            
        try:
            response = self._client.embeddings.create(
                input=[text],
                model=self._model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Embedding 錯誤: {e}")
            return []
    
    def get_embeddings_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """批次取得 embeddings
        
        Args:
            texts: 文字列表
            batch_size: 每批數量（OpenAI 限制）
            
        Returns:
            向量嵌入列表
        """
        all_embeddings: list[list[float]] = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # 清理文字
            batch = [t.replace("\n", " ").strip() for t in batch]
            
            try:
                response = self._client.embeddings.create(
                    input=batch,
                    model=self._model
                )
                batch_embeddings = [d.embedding for d in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"❌ 批次 Embedding 錯誤: {e}")
                # 填充空向量
                all_embeddings.extend([[] for _ in batch])
        
        return all_embeddings
