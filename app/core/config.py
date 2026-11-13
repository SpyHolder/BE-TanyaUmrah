import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "tanyaUMRAH Backend API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Gemini AI
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    FALLBACK_MODEL: str = "gemini-3.1-flash-lite"
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    
    # Database & Storage
    DATABASE_URL: str = "sqlite:///./tanyaumrah.db"
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    UPLOAD_DIR: str = "./storage/uploads"
    DATA_PDF_DIR: str = r"C:\HOLDER\KULIAH\Semester 5\PPB\data"
    
    # Admin Credentials (Local Testing)
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "admin123"
    JWT_SECRET: str = "tanyaumrah_local_secret_key_123456"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24  # 1 day

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
