"""
Centralized application configuration.

Semua konstanta & setting yang bisa berubah (jumlah kursi, threshold engagement,
ukuran window temporal, dsb) ditaruh di sini — BUKAN hardcode di file lain —
supaya gampang diubah tanpa nyari-nyari di banyak file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- App Info ---
    app_name: str = "Axon Classroom Engagement Analytics"

    # --- Kamera / Seat Management ---
    valid_seat_ids: list[str] = [
    "A1", "A2", "A3", "A4", "A5", "A6",
    "B1", "B2", "B3", "B4", "B5", "B6",
    "C1", "C2", "C3", "C4", "C5", "C6",
    "D1", "D2", "D3", "D4", "D5", "D6",
    "E1", "E2", "E3", "E4", "E5", "E6",
]
    # --- Concurrency Control ---
    max_concurrent_inference: int = 2   

    # --- Temporal Analysis ---
    temporal_window_size: int = 10
    low_engagement_consecutive_threshold: int = 5

    # --- Drowsiness Detection (EAR) ---
    # Threshold rasio bukaan mata. Nilai umum di riset = 0.21-0.25,
    # TAPI WAJIB dikalibrasi ulang manual pakai foto asli dari ESP32-CAM
    # kelas lo (angle kamera beda = threshold ideal bisa beda).
    ear_threshold: float = 0.21
    # Berapa frame berturut-turut EAR rendah sebelum status jadi "Drowsy".
    # Dipisah dari low_engagement_consecutive_threshold karena secara
    # konsep beda (ngantuk fisiologis vs disengagement emosional).
    ear_consecutive_frames: int = 5

    # --- Distraction Detection (Head Pose / Yaw) ---
    # Berapa derajat dianggap "menengok" dari kamera. Nilai awal perkiraan,
    # WAJIB dikalibrasi manual pakai foto asli dari ESP32-CAM kelas lo.
    yaw_threshold_degrees: float = 25.0
    # Sama filosofinya kayak ear_consecutive_frames -- butuh konsisten
    # beberapa frame biar nengok sebentar (misal liat papan tulis) gak
    # langsung dianggap "gak fokus".
    yaw_consecutive_frames: int = 5

    # --- Model Paths ---
    emotiefflib_model_name: str = "enet_b0_8_best_afew"
    face_landmarker_model_name: str = "face_landmarker.task"

    # --- Firebase ---
    firebase_credentials_path: str = "app/firebase/service_account.json"
    firestore_collection_name: str = "engagement_logs"


settings = Settings()