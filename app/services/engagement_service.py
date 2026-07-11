"""
Service orkestrasi utama: dari foto mentah sampai hasil akhir tersimpan.

Concurrency control (asyncio.Semaphore) ada di sini karena inference
(MediaPipe + EmotiEffLib) itu CPU-bound. Semaphore membatasi berapa
banyak inference yang boleh jalan BERSAMAAN, sesuai settings.max_concurrent_inference,
supaya 5 ESP32-CAM yang request hampir bersamaan tidak membanjiri CPU.
"""

import asyncio
import io

import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.rule_engine import RuleEngine
from app.firebase.client import save_engagement_log
from app.models.emotion_recognizer import EmotionRecognizerService
from app.models.face_detector import FaceDetectorService
from app.schemas.prediction import EngagementLogEntry, PredictionResponse, utc_now

# --- Singleton instances ---
# Model di-load SEKALI saat modul ini pertama kali diimport (saat server
# start), BUKAN setiap kali ada request masuk — supaya tidak reload model
# yang mahal secara berulang.
_face_detector = FaceDetectorService()
_emotion_recognizer = EmotionRecognizerService()
_rule_engine = RuleEngine()

# Semaphore membatasi jumlah inference yang berjalan bersamaan.
_inference_semaphore = asyncio.Semaphore(settings.max_concurrent_inference)


async def process_frame(image_bytes: bytes, seat: str) -> PredictionResponse:
    """
    Alur lengkap: decode foto -> deteksi wajah -> prediksi emosi ->
    hitung engagement -> simpan ke Firestore -> return response.
    """
    image_rgb = _decode_image(image_bytes)

    async with _inference_semaphore:
        # Inference (MediaPipe + EmotiEffLib) itu blocking/sync, jadi
        # dijalankan di thread terpisah (run_in_executor) supaya tidak
        # nge-block event loop FastAPI selagi CPU sibuk memproses.
        face_crop = await asyncio.get_event_loop().run_in_executor(
            None, _face_detector.detect_and_crop, image_rgb
        )

        if face_crop is None:
            return PredictionResponse(
                seat=seat,
                emotion="none",
                confidence=0.0,
                engagement_score=0,
                status="No Face Detected",
                timestamp=utc_now(),
            )

        emotion, confidence = await asyncio.get_event_loop().run_in_executor(
            None, _emotion_recognizer.predict, face_crop
        )

    engagement_score, status = _rule_engine.compute_engagement(seat, emotion, confidence)

    response = PredictionResponse(
        seat=seat,
        emotion=emotion,
        confidence=confidence,
        engagement_score=engagement_score,
        status=status,
        timestamp=utc_now(),
    )

    # Simpan ke Firestore (fire-and-forget secara logis, tapi tetap
    # di-await supaya error penulisan DB tidak hilang diam-diam)
    log_entry = EngagementLogEntry.from_prediction(response)
    await asyncio.get_event_loop().run_in_executor(None, save_engagement_log, log_entry)

    return response


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode bytes JPEG/PNG dari upload jadi numpy array RGB (HWC, uint8)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)