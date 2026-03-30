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
    COLD_STORE: str = os.getenv("GAMI_COLD_STORE", "/mnt/16tb/gami")

    # Logging
    LOG_LEVEL: str = os.getenv("GAMI_LOG_LEVEL", "INFO")

settings = Settings()
