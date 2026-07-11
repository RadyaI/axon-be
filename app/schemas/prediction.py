"""
Pydantic schemas untuk validasi request masuk & response keluar
di endpoint /predict.

Kenapa pakai schema terpisah dari model ML:
- Schema ini adalah "kontrak" HTTP API, bisa berubah sesuai kebutuhan frontend.
- Model ML (di app/models/) punya representasi internal sendiri.
- Memisahkan keduanya supaya perubahan di satu sisi gak otomatis
  merusak sisi lain (lebih maintainable).
"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings

# Status akhir yang mungkin dikeluarkan Rule Engine.
EngagementStatus = Literal["Focused", "Neutral", "Need Attention", "No Face Detected"]


class PredictionResponse(BaseModel):
    """Response yang dikembalikan endpoint /predict ke ESP32-CAM / caller."""

    seat: str
    emotion: str
    confidence: float = Field(ge=0.0, le=1.0)
    engagement_score: int = Field(ge=0, le=100)
    status: EngagementStatus
    timestamp: datetime

    @field_validator("seat")
    @classmethod
    def validate_seat(cls, v: str) -> str:
        if v not in settings.valid_seat_ids:
            raise ValueError(
                f"seat '{v}' tidak dikenal. Seat valid: {settings.valid_seat_ids}"
            )
        return v


class EngagementLogEntry(BaseModel):
    """
    Representasi data yang DISIMPAN ke Firestore.
    Sengaja dipisah dari PredictionResponse walau isinya mirip,
    karena field yang disimpan ke DB bisa saja berbeda dari yang
    dikembalikan ke client (misal nanti mau nambah field internal
    tanpa expose ke response API).
    """

    seat: str
    emotion: str
    confidence: float
    engagement_score: int
    status: EngagementStatus
    timestamp: datetime

    @staticmethod
    def from_prediction(prediction: PredictionResponse) -> "EngagementLogEntry":
        return EngagementLogEntry(**prediction.model_dump())


def utc_now() -> datetime:
    """Helper biar semua timestamp konsisten pakai UTC (hindari bug timezone)."""
    return datetime.now(timezone.utc)