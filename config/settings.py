from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API keys
    anthropic_api_key: str
    jina_api_key: str

    # Infrastructure
    postgres_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/ragdb"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None   # required for Qdrant Cloud, None for local

    # Models
    enrichment_model: str = "claude-haiku-4-5-20251001"
    generation_model: str = "claude-sonnet-4-6"
    jina_embed_model: str = "jina-embeddings-v3"
    jina_image_model: str = "jina-embeddings-v4"
    jina_rerank_model: str = "jina-reranker-v3"
    embed_dimensions: int = 1024

    # Retrieval
    top_k_retrieval: int = 20
    top_n_reranked: int = 5
    reranker_score_threshold: float = 0.3
    semantic_cache_threshold: float = 0.95

    # Ingestion
    chunk_batch_size: int = 20
    table_store_dir: str = "./table_store"
    upload_dir: str = "./uploads"


settings = Settings()
