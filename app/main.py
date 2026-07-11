"""
Entry point aplikasi FastAPI.

Jalankan dengan (dari root folder axon, bukan dari dalam app/):
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.predict import router as predict_router
from app.core.config import settings
from app.firebase.client import init_firebase


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: inisialisasi Firebase sekali sebelum server mulai terima request.
    init_firebase()
    yield
    # Shutdown: (belum ada cleanup khusus yang dibutuhkan untuk saat ini)


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.include_router(predict_router)


@app.get("/")
async def health_check():
    """Endpoint sederhana buat cek server hidup atau tidak."""
    return {"status": "ok", "app": settings.app_name}