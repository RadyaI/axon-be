"""
Firebase Admin SDK client — inisialisasi sekali (singleton), dipakai
untuk menulis data engagement ke Firestore.

PENTING (privasi): fungsi di sini HANYA menulis data numerik/label
(EngagementLogEntry), TIDAK PERNAH menyimpan foto/gambar apa pun.
"""

import firebase_admin
from firebase_admin import credentials, firestore

from app.core.config import settings
from app.schemas.prediction import EngagementLogEntry

_app: firebase_admin.App | None = None
_db = None


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