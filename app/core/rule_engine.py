"""
Rule Engine -- mengubah (emotion, confidence, ear) mentah dari AI
menjadi engagement_score (0-100) dan status.
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
        # History EAR terpisah dari engagement_score, karena "ngantuk"
        # dan "disengagement emosional" itu 2 sinyal berbeda yang perlu
        # dilacak konsistensinya masing-masing.
        self._ear_history: dict[str, list[float]] = {}

    def compute_engagement(
        self, seat: str, emotion: str, confidence: float, ear: float
    ) -> tuple[int, str]:
        raw_score = self._emotion_to_score(emotion, confidence)
        self._update_history(seat, raw_score)
        self._update_ear_history(seat, ear)

        smoothed_score = self._smoothed_score(seat)
        is_drowsy = self._is_consistently_drowsy(seat)
        status = self._determine_status(seat, smoothed_score, is_drowsy)

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
        # Jaga history EAR gak lebih panjang dari yang dibutuhkan buat
        # cek consecutive frame -- gak perlu ikut temporal_window_size.
        max_len = settings.ear_consecutive_frames
        if len(history) > max_len:
            history.pop(0)

    def _smoothed_score(self, seat: str) -> int:
        history = self._history[seat]
        return round(sum(history) / len(history))

    def _is_consistently_drowsy(self, seat: str) -> bool:
        """
        True kalau EAR di bawah threshold selama N frame berturut-turut
        (N = ear_consecutive_frames). Sama filosofinya kayak
        low_engagement_consecutive_threshold -- gak boleh dari 1 frame
        doang, biar kedipan mata normal gak dianggap ngantuk.
        """
        history = self._ear_history.get(seat, [])
        threshold = settings.ear_consecutive_frames

        if len(history) < threshold:
            return False

        return all(ear < settings.ear_threshold for ear in history[-threshold:])

    def _determine_status(self, seat: str, smoothed_score: int, is_drowsy: bool) -> str:
        # Cek drowsy DULUAN -- ini override status lain, karena mata
        # tertutup konsisten adalah sinyal fisiologis yang lebih kuat
        # dibanding skor emosi buat kasus "tidur di meja".
        if is_drowsy:
            return "Drowsy"

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