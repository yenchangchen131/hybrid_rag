# Core Module
"""核心基礎設施模組：配置、資料庫連線、日誌"""

from core.config import settings
from core.database import MongoDBClient

__all__ = ["settings", "MongoDBClient"]
