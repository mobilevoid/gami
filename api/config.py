import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Database
    DB_HOST: str = os.getenv("GAMI_DB_HOST", "127.0.0.1")
    DB_PORT: int = int(os.getenv("GAMI_DB_PORT", "5433"))
    DB_NAME: str = os.getenv("GAMI_DB_NAME", "gami")
    DB_USER: str = os.getenv("GAMI_DB_USER", "gami")
    DB_PASSWORD: str = os.getenv("GAMI_DB_PASSWORD", "")

    @property
    def DATABASE_URL(self) -> str:
        pw = quote_plus(self.DB_PASSWORD)
        return f"postgresql+asyncpg://{self.DB_USER}:{pw}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def DATABASE_URL_SYNC(self) -> str:
        pw = quote_plus(self.DB_PASSWORD)
        return f"postgresql+psycopg2://{self.DB_USER}:{pw}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # Redis
    REDIS_HOST: str = os.getenv("GAMI_REDIS_HOST", "127.0.0.1")
    REDIS_PORT: int = int(os.getenv("GAMI_REDIS_PORT", "6380"))
    REDIS_DB: int = int(os.getenv("GAMI_REDIS_DB", "0"))

    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # LLM
    VLLM_URL: str = os.getenv("GAMI_VLLM_URL", "http://localhost:8000/v1")
    OLLAMA_URL: str = os.getenv("GAMI_OLLAMA_URL", "http://localhost:11434")
    EMBEDDING_MODEL: str = os.getenv("GAMI_EMBEDDING_MODEL", "nomic-embed-text")
    EXTRACTION_MODEL: str = os.getenv("GAMI_EXTRACTION_MODEL", "qwen35-27b-unredacted")
    CLASSIFICATION_MODEL: str = os.getenv("GAMI_CLASSIFICATION_MODEL", "qwen3:8b")

    # API
    API_HOST: str = os.getenv("GAMI_API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("GAMI_API_PORT", "9000"))
    MCP_PORT: int = int(os.getenv("GAMI_MCP_PORT", "9001"))

    # Storage
    OBJECT_STORE: str = os.getenv("GAMI_OBJECT_STORE", "/opt/gami/storage/objects")
    COLD_STORE: str = os.getenv("GAMI_COLD_STORE", "/var/lib/gami/cold")

    # Logging
    LOG_LEVEL: str = os.getenv("GAMI_LOG_LEVEL", "INFO")

    # Authentication (optional - set GAMI_API_KEY to enable)
    API_KEY: str = os.getenv("GAMI_API_KEY", "")  # Empty = auth disabled
    REQUIRE_AUTH_FOR_AGENTS: bool = os.getenv("GAMI_REQUIRE_AUTH_FOR_AGENTS", "false").lower() == "true"

    # Cross-Encoder Reranking
    RERANKER_ENABLED: bool = os.getenv("GAMI_RERANKER_ENABLED", "true").lower() == "true"
    RERANKER_MODEL: str = os.getenv("GAMI_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_TOP_K: int = int(os.getenv("GAMI_RERANKER_TOP_K", "50"))
    RERANKER_FINAL_K: int = int(os.getenv("GAMI_RERANKER_FINAL_K", "15"))
    RERANKER_BLEND_RATIO: float = float(os.getenv("GAMI_RERANKER_BLEND_RATIO", "0.7"))

    # Memory Consolidation (Mem0-style)
    MEMORY_CONSOLIDATION_ENABLED: bool = os.getenv("GAMI_MEMORY_CONSOLIDATION_ENABLED", "true").lower() == "true"
    MEMORY_SIMILAR_THRESHOLD: float = float(os.getenv("GAMI_MEMORY_SIMILAR_THRESHOLD", "0.75"))
    MEMORY_EXACT_THRESHOLD: float = float(os.getenv("GAMI_MEMORY_EXACT_THRESHOLD", "0.95"))


settings = Settings()
