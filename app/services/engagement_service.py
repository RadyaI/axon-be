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

_face_detector = FaceDetectorService()
_emotion_recognizer = EmotionRecognizerService()
_rule_engine = RuleEngine()

_inference_semaphore = asyncio.Semaphore(settings.max_concurrent_inference)


async def process_frame(image_bytes: bytes, seat: str) -> PredictionResponse:
    image_rgb = _decode_image(image_bytes)

    async with _inference_semaphore:
        detection = await asyncio.get_event_loop().run_in_executor(
            None, _face_detector.detect, image_rgb
        )

        if detection is None:
            return PredictionResponse(
                seat=seat,
                emotion="none",
                confidence=0.0,
                ear=None,
                engagement_score=0,
                status="No Face Detected",
                timestamp=utc_now(),
            )

        face_crop, ear = detection

        emotion, confidence = await asyncio.get_event_loop().run_in_executor(
            None, _emotion_recognizer.predict, face_crop
        )

    engagement_score, status = _rule_engine.compute_engagement(seat, emotion, confidence, ear)

    response = PredictionResponse(
        seat=seat,
        emotion=emotion,
        confidence=confidence,
        ear=round(ear, 4),
        engagement_score=engagement_score,
        status=status,
        timestamp=utc_now(),
    )

    log_entry = EngagementLogEntry.from_prediction(response)
    await asyncio.get_event_loop().run_in_executor(None, save_engagement_log, log_entry)

    return response


def _decode_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)