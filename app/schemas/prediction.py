from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings

EngagementStatus = Literal["Focused", "Neutral", "Need Attention", "Drowsy", "No Face Detected"]


class PredictionResponse(BaseModel):
    seat: str
    emotion: str
    confidence: float = Field(ge=0.0, le=1.0)
    ear: float | None = Field(default=None, ge=0.0)
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
    seat: str
    emotion: str
    confidence: float
    ear: float | None = None
    engagement_score: int
    status: EngagementStatus
    timestamp: datetime

    @staticmethod
    def from_prediction(prediction: PredictionResponse) -> "EngagementLogEntry":
        return EngagementLogEntry(**prediction.model_dump())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class EngagementListResponse(BaseModel):
    count: int
    data: list[EngagementLogEntry]


class SeatLatestEntry(BaseModel):
    seat: str
    data: EngagementLogEntry | None  # null kalau seat itu belum pernah kirim data


class LatestEngagementResponse(BaseModel):
    data: list[SeatLatestEntry]