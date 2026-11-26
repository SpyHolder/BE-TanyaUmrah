import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.database.session import Base, engine
from app.routers import admin, auth, calendar, chat, documents

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("tanyaUMRAH")


@asynccontextmanager
async def lifespan(app):
    Base.metadata.create_all(bind=engine)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    yield


app = FastAPI(
    lifespan=lifespan,
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API & RAG Engine for tanyaUMRAH Student Assistant",
)

# CORS setup for Flutter & Web Admin
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API v1 Routers
app.include_router(chat.router, prefix=settings.API_V1_PREFIX)
app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(documents.router, prefix=settings.API_V1_PREFIX)
app.include_router(calendar.router, prefix=settings.API_V1_PREFIX)

# Include Admin Panel Router
app.include_router(admin.router, prefix="/admin")


@app.get("/")
def root():
    return {
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs_url": "/docs",
        "admin_url": "/admin",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
