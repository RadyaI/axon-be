"""
Rule Engine -- mengubah (emotion, confidence, ear, yaw) mentah dari AI
menjadi engagement_score (0-100) dan status.

Prioritas status (dari paling "yakin" ke paling "soft"):
  1. Drowsy       -- EAR rendah konsisten (sinyal fisiologis paling jelas)
  2. Distracted   -- yaw tinggi konsisten (menengok, tapi mata melek)
  3. Need Attention -- engagement_score rendah konsisten
  4. Focused      -- engagement_score tinggi
  5. Neutral      -- default
"""

from app.core.config import settings

EMOTION_WEIGHTS: dict[str, int] = {
    "Happiness": 85,
    "Neutral": 75,
    "Surprise": 60,
    "Contempt": 45,
    "Sadness": 30,
    "Fear": 25,
    "Anger": 20,
    "Disgust": 20,
}

DEFAULT_WEIGHT_IF_UNKNOWN = 50


class RuleEngine:
    def __init__(self) -> None:
        self._history: dict[str, list[int]] = {}
        self._ear_history: dict[str, list[float]] = {}
        self._yaw_history: dict[str, list[float]] = {}

    def compute_engagement(
        self, seat: str, emotion: str, confidence: float, ear: float, yaw: float
    ) -> tuple[int, str]:
        raw_score = self._emotion_to_score(emotion, confidence)
        self._update_history(seat, raw_score)
        self._update_ear_history(seat, ear)
        self._update_yaw_history(seat, yaw)

        smoothed_score = self._smoothed_score(seat)
        is_drowsy = self._is_consistently_drowsy(seat)
        is_distracted = self._is_consistently_distracted(seat)
        status = self._determine_status(seat, smoothed_score, is_drowsy, is_distracted)

        return smoothed_score, status

    def _emotion_to_score(self, emotion: str, confidence: float) -> int:
        base_weight = EMOTION_WEIGHTS.get(emotion, DEFAULT_WEIGHT_IF_UNKNOWN)
        neutral_point = 50
        adjusted = neutral_point + (base_weight - neutral_point) * confidence
        return round(adjusted)

    def _update_history(self, seat: str, score: int) -> None:
        history = self._history.setdefault(seat, [])
        history.append(score)
        if len(history) > settings.temporal_window_size:
            history.pop(0)

    def _update_ear_history(self, seat: str, ear: float) -> None:
        history = self._ear_history.setdefault(seat, [])
        history.append(ear)
        if len(history) > settings.ear_consecutive_frames:
            history.pop(0)

    def _update_yaw_history(self, seat: str, yaw: float) -> None:
        history = self._yaw_history.setdefault(seat, [])
        history.append(yaw)
        if len(history) > settings.yaw_consecutive_frames:
            history.pop(0)

    def _smoothed_score(self, seat: str) -> int:
        history = self._history[seat]
        return round(sum(history) / len(history))

    def _is_consistently_drowsy(self, seat: str) -> bool:
        history = self._ear_history.get(seat, [])
        threshold = settings.ear_consecutive_frames

        if len(history) < threshold:
            return False

        return all(ear < settings.ear_threshold for ear in history[-threshold:])

    def _is_consistently_distracted(self, seat: str) -> bool:
        """
        True kalau |yaw| di atas threshold selama N frame berturut-turut.
        Pakai nilai absolut karena "nengok kiri" dan "nengok kanan"
        sama-sama berarti tidak menghadap kamera/guru.
        """
        history = self._yaw_history.get(seat, [])
        threshold = settings.yaw_consecutive_frames

        if len(history) < threshold:
            return False

        return all(
            abs(yaw) > settings.yaw_threshold_degrees for yaw in history[-threshold:]
        )

    def _determine_status(
        self, seat: str, smoothed_score: int, is_drowsy: bool, is_distracted: bool
    ) -> str:
        # Drowsy paling prioritas -- mata merem konsisten adalah sinyal
        # paling tidak ambigu, override semua sinyal lain.
        if is_drowsy:
            return "Drowsy"

        # Distracted dicek SETELAH drowsy -- kalau ternyata mata juga
        # merem, itu lebih tepat diklasifikasikan sebagai Drowsy, bukan
        # Distracted (menengok bukan penyebab utamanya).
        if is_distracted:
            return "Distracted"

        history = self._history[seat]
        threshold = settings.low_engagement_consecutive_threshold
        recent_frames = history[-threshold:]
        is_consistently_low = (
            len(recent_frames) >= threshold
            and all(score < 50 for score in recent_frames)
        )

        if is_consistently_low:
            return "Need Attention"
        if smoothed_score >= 65:
            return "Focused"
        return "Neutral"

    def reset_seat(self, seat: str) -> None:
        self._history.pop(seat, None)
        self._ear_history.pop(seat, None)
        self._yaw_history.pop(seat, None)