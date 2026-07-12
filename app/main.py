"""
Entry point aplikasi FastAPI.
    uvicorn app.main:app --reload

"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.engagement import router as engagement_router
from app.api.predict import router as predict_router
from app.api.session import router as session_router
from app.core.config import settings
from app.firebase.client import init_firebase


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_firebase()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

# CORS -- wajib ada supaya frontend (yang jalan di origin/domain berbeda,
# misal localhost:5173 buat Vite atau localhost:3000 buat Next.js) bisa
# fetch ke backend ini. Tanpa ini, browser bakal blokir request-nya
# duluan sebelum sempat nyampai ke endpoint manapun.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # sementara buat development, lihat catatan di bawah
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(session_router)
app.include_router(engagement_router)


@app.get("/")
async def health_check():
    return {"status": "ok", "app": settings.app_name}