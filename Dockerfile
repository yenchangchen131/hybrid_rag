# Hybrid RAG Docker Image
# 
# Build: docker build -t hybrid-rag .
# Run:   docker run --env-file .env hybrid-rag

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# 複製依賴檔案
COPY pyproject.toml uv.lock ./

# 安裝依賴 (不含開發依賴)
RUN uv sync --frozen --no-dev

# 複製程式碼
COPY core/ ./core/
COPY models/ ./models/
COPY repositories/ ./repositories/
COPY services/ ./services/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY main.py ./

# 預設命令
CMD ["uv", "run", "python", "main.py"]
