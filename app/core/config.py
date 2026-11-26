from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    PROJECT_NAME: str = "tanyaUMRAH Backend API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    FALLBACK_MODEL: str = "gemini-3.1-flash-lite"
    EMBEDDING_MODEL: str = "local"
    RETRIEVAL_MIN_SIMILARITY: float = Field(0.30, ge=0, le=1)

    DATABASE_URL: str = f"sqlite:///{(BACKEND_DIR / 'tanyaumrah.db').as_posix()}"
    CHROMA_PERSIST_DIR: str = str(BACKEND_DIR / "chroma_data")
    UPLOAD_DIR: str = str(BACKEND_DIR / "storage" / "uploads")
    CACHE_TEXT_DIR: str = str(BACKEND_DIR / "storage" / "cache_text")
    DATA_PDF_DIR: str = str(
        (BACKEND_DIR / "data")
        if (BACKEND_DIR / "data").exists()
        else (BACKEND_DIR.parent / "data")
    )
    MAX_UPLOAD_BYTES: int = Field(20 * 1024 * 1024, ge=1024)

    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    FIREBASE_CREDENTIALS_PATH: str = ""
    FIREBASE_PROJECT_ID: str = ""
    ADMIN_USERNAME: str = ""
    ADMIN_PASSWORD: str = ""
    # JSON array in .env; add the fixed Flutter web origin when needed.
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]


settings = Settings()
