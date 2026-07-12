"""
Endpoint HTTP untuk MEMBACA data engagement dari Firestore.

Dipakai oleh dashboard (frontend) -- beda dengan predict.py yang
menerima tulisan data dari ESP32-CAM, file ini murni baca data.
"""

from fastapi import APIRouter, Query

from app.core.config import settings
from app.firebase.client import get_engagement_logs, get_latest_per_seat
from app.schemas.prediction import (
    EngagementListResponse,
    LatestEngagementResponse,
    SeatLatestEntry,
)

router = APIRouter(prefix="/engagement", tags=["engagement"])


@router.get("", response_model=EngagementListResponse)
async def list_engagement(
    limit: int = Query(default=50, ge=1, le=200),
    seat: str | None = Query(default=None),
) -> EngagementListResponse:
    """
    Riwayat log engagement, urut dari yang terbaru.
    Contoh: GET /engagement?limit=20&seat=A1
    """
    logs = get_engagement_logs(limit=limit, seat=seat)
    return EngagementListResponse(count=len(logs), data=logs)


@router.get("/latest", response_model=LatestEngagementResponse)
async def latest_engagement() -> LatestEngagementResponse:
    """
    Snapshot kondisi kelas saat ini -- 1 data terbaru PER kursi.
    Kursi yang belum pernah kirim data tetap muncul (data=null),
    biar dashboard tetap render 5 kartu kursi walau ada yang kosong.
    """
    latest_map = get_latest_per_seat()
    entries = [
        SeatLatestEntry(seat=seat, data=latest_map[seat])
        for seat in settings.valid_seat_ids
    ]
    return LatestEngagementResponse(data=entries)