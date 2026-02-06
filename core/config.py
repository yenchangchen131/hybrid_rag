"""
統一配置管理

使用 Pydantic Settings 從環境變數載入配置，支援 .env 檔案。
"""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """應用程式配置"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # MongoDB 配置
    MONGODB_HOST: str = "localhost"
    MONGODB_PORT: int = 27017
    MONGODB_DB_NAME: str = "hybrid_rag"
    COLLECTION_NAME: str = "documents"
    
    # OpenAI 配置
    OPENAI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.7
    
    # 檢索配置
    DEFAULT_TOP_K: int = 5
    INITIAL_RETRIEVAL_K: int = 20  # 初始檢索數量（用於 RRF 融合前）
    RRF_K: int = 60  # RRF 融合參數
    
    # 路徑配置
    DATA_DIR: Path = Path("data")
    
    @property
    def mongodb_uri(self) -> str:
        """MongoDB 連線 URI"""
        return f"mongodb://{self.MONGODB_HOST}:{self.MONGODB_PORT}/"
    
    @property
    def corpus_path(self) -> Path:
        """文件庫路徑"""
        return self.DATA_DIR / "corpus.json"
    
    @property
    def queries_path(self) -> Path:
        """測試查詢路徑"""
        return self.DATA_DIR / "queries.json"


@lru_cache
def get_settings() -> Settings:
    """取得配置單例"""
    return Settings()


# 全域配置實例
settings = get_settings()
