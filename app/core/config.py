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
    # Daftar seat_id yang valid untuk prototype 5 ESP32-CAM.
    # Kalau nanti nambah kamera, cukup update list ini.
    valid_seat_ids: list[str] = ["A1", "A2", "A3", "A4", "A5"]

    # --- Concurrency Control ---
    # Batas jumlah inference (MediaPipe + EmotiEffLib) yang boleh jalan
    # BERSAMAAN di CPU. Karena inference itu CPU-bound & laptop biasa
    # gak punya banyak core buat paralel penuh, kita batasi supaya
    # gak saling rebutan resource dan bikin latency melonjak.
    # Mulai dari 1 dulu (paling aman/stabil), bisa dinaikkan kalau CPU kuat.
    max_concurrent_inference: int = 1

    # --- Temporal Analysis ---
    # Jumlah frame historis yang disimpan per seat untuk smoothing
    # & deteksi tren (bukan keputusan dari 1 frame doang).
    temporal_window_size: int = 10

    # Berapa frame berturut-turut engagement rendah sebelum status
    # berubah jadi "Need Attention".
    low_engagement_consecutive_threshold: int = 5

    # --- Model Paths ---
    emotiefflib_model_name: str = "enet_b0_8_best_afew"  # model ringan, cocok CPU

    # --- Firebase ---
    firebase_credentials_path: str = "app/firebase/service_account.json"
    firestore_collection_name: str = "engagement_logs"


settings = Settings()