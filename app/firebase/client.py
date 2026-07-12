"""
Firebase Admin SDK client — inisialisasi sekali (singleton), dipakai
untuk menulis data engagement ke Firestore.

PENTING (privasi): fungsi di sini HANYA menulis data numerik/label
(EngagementLogEntry), TIDAK PERNAH menyimpan foto/gambar apa pun.
"""

from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore

from app.core.config import settings
from app.schemas.prediction import EngagementLogEntry

_app: firebase_admin.App | None = None
_db = None

# Session status disimpan di 1 dokumen tunggal (bukan collection log),
# karena statusnya cuma 1 untuk semua kelas -- bukan riwayat per-event.
_SESSION_COLLECTION = "system_state"
_SESSION_DOCUMENT = "session"


def init_firebase() -> None:
    """Dipanggil sekali saat FastAPI startup (lihat app/main.py)."""
    global _app, _db
    if _app is not None:
        return  # sudah diinisialisasi, hindari init dobel

    cred = credentials.Certificate(settings.firebase_credentials_path)
    _app = firebase_admin.initialize_app(cred)
    _db = firestore.client()


def save_engagement_log(entry: EngagementLogEntry) -> None:
    """Simpan 1 entry log ke Firestore collection yang dikonfigurasi."""
    if _db is None:
        raise RuntimeError("Firebase belum diinisialisasi. Panggil init_firebase() dulu.")

    _db.collection(settings.firestore_collection_name).add(entry.model_dump())


def get_session_status() -> bool:
    """
    Baca status sesi kelas saat ini (aktif/tidak).

    Kalau dokumen belum pernah dibuat sama sekali (server baru pertama
    kali dijalankan), default-nya FALSE -- lebih aman, ESP32-CAM tidak
    kirim foto sampai guru sengaja menyalakan sesi lewat dashboard.
    """
    if _db is None:
        raise RuntimeError("Firebase belum diinisialisasi. Panggil init_firebase() dulu.")

    doc = _db.collection(_SESSION_COLLECTION).document(_SESSION_DOCUMENT).get()

    if not doc.exists:
        return False

    return doc.to_dict().get("active", False)


def set_session_status(active: bool) -> None:
    """Ubah status sesi kelas, dipanggil dari endpoint /session/start atau /stop."""
    if _db is None:
        raise RuntimeError("Firebase belum diinisialisasi. Panggil init_firebase() dulu.")

    _db.collection(_SESSION_COLLECTION).document(_SESSION_DOCUMENT).set({
        "active": active,
        "updated_at": datetime.now(timezone.utc),
    })

def get_engagement_logs(limit: int = 50, seat: str | None = None) -> list[dict]:
    """Ambil riwayat log, urut dari yang terbaru. Opsional filter per seat."""
    if _db is None:
        raise RuntimeError("Firebase belum diinisialisasi. Panggil init_firebase() dulu.")

    query = _db.collection(settings.firestore_collection_name)
    if seat:
        query = query.where("seat", "==", seat)
    query = query.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit)

    return [doc.to_dict() for doc in query.stream()]


def get_latest_per_seat() -> dict[str, dict | None]:
    """
    Ambil 1 dokumen terbaru untuk TIAP seat (query terpisah per seat,
    bukan narik semua data lalu difilter manual -- lebih efisien
    karena Firestore sendiri yang menyaring, bukan Python).
    """
    if _db is None:
        raise RuntimeError("Firebase belum diinisialisasi. Panggil init_firebase() dulu.")

    result: dict[str, dict | None] = {}

    for seat in settings.valid_seat_ids:
        docs = list(
            _db.collection(settings.firestore_collection_name)
            .where("seat", "==", seat)
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        result[seat] = docs[0].to_dict() if docs else None

    return result