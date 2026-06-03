FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils curl libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Inline all dependencies — avoids build context issues on Railway
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.115.12 \
    "uvicorn[standard]==0.34.3" \
    python-multipart==0.0.20 \
    sse-starlette==2.3.6 \
    pydantic-settings==2.9.1 \
    pydantic==2.11.5 \
    anthropic==0.52.0 \
    httpx==0.28.1 \
    tenacity==9.1.2 \
    "qdrant-client[async]==1.14.2" \
    "sqlalchemy[asyncio]==2.0.41" \
    asyncpg==0.30.0 \
    psycopg2-binary==2.9.10 \
    "celery[redis]==5.5.2" \
    redis==5.3.0 \
    pandas==2.2.3 \
    openpyxl==3.1.5 \
    pyarrow==20.0.0 \
    asteval==1.0.5 \
    pymupdf==1.24.14 \
    pypdf==5.1.0 \
    python-docx==1.1.2 \
    python-pptx==1.0.2 \
    python-dotenv==1.1.0 \
    structlog==25.3.0 \
    pillow==11.2.1

COPY . .
RUN chmod +x start.sh

EXPOSE 8000
CMD ["bash", "start.sh"]
