"""
統一日誌配置
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """設定應用程式日誌
    
    Args:
        level: 日誌級別
        
    Returns:
        配置好的 logger
    """
    logger = logging.getLogger("hybrid_rag")
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # 格式
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


# 預設 logger
logger = setup_logging()
