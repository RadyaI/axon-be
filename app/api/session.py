"""
Endpoint HTTP untuk kontrol sesi kelas (on/off).

Dipakai oleh 2 pihak:
- Dashboard (guru): POST /session/start dan /session/stop, untuk
  menyalakan/mematikan pengambilan data secara terpusat.
- ESP32-CAM: GET /session/status, dipanggil berkala sebelum mengambil
  foto -- kalau sesi tidak aktif, device TIDAK mengirim foto sama sekali.

File ini sengaja TIPIS (thin layer), sama seperti app/api/predict.py --
logic penyimpanan status ada di app/firebase/client.py, dan logic reset
histori rule engine ada di app/services/engagement_service.py.
"""

from fastapi import APIRouter

from app.firebase.client import get_session_status, set_session_status
from app.services.engagement_service import reset_all_seats

router = APIRouter(prefix="/session", tags=["session"])


@router.get("/status")
async def session_status() -> dict:
    """Dipanggil ESP32-CAM secara berkala untuk cek apakah boleh kirim foto."""
    return {"active": get_session_status()}


@router.post("/start")
async def start_session() -> dict:
    """
    Mulai sesi baru.

    Sekalian reset histori rule engine (engagement_score & EAR) untuk
    semua seat -- supaya sesi baru tidak "terkontaminasi" sisa histori
    dari sesi sebelumnya yang masih nempel di RAM.
    """
    set_session_status(True)
    reset_all_seats()
    return {"active": True}


@router.post("/stop")
async def stop_session() -> dict:
    """Hentikan sesi. ESP32-CAM akan berhenti mengirim foto begitu cek status berikutnya."""
    set_session_status(False)
    return {"active": False}