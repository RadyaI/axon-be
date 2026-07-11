"""
Rule Engine — mengubah (emotion, confidence) mentah dari AI menjadi
engagement_score (0-100) dan status ("Focused" / "Neutral" / "Need Attention").

Desain sengaja config-driven (EMOTION_WEIGHTS sebagai dict), BUKAN
if-elif menumpuk, supaya:
1. Bobot tiap emosi gampang di-tuning tanpa bongkar logic.
2. Nambah/ubah emosi baru tinggal ubah dict, gak perlu sentuh method lain.
"""

from app.core.config import settings

# --- Bobot dasar tiap emosi terhadap engagement (skala 0-100) ---
# Ini ASUMSI AWAL yang MASIH BISA di-tuning berdasarkan observasi lapangan
# nanti. Logikanya:
#   - Neutral & Happiness -> asosiasi dengan siswa yang engaged/fokus
#   - Sadness, Fear, Anger, Disgust -> asosiasi dengan distraksi/disengagement
#   - Surprise, Contempt -> netral cenderung ambigu, diberi skor tengah
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

DEFAULT_WEIGHT_IF_UNKNOWN = 50  # fallback kalau ada label emosi yang gak dikenal


class RuleEngine:
    """
    Mengelola state temporal per seat (histori beberapa frame terakhir)
    dan menentukan status akhir berdasarkan tren, bukan 1 frame doang.
    """

    def __init__(self) -> None:
        # Histori engagement_score per seat, contoh:
        # {"A1": [70, 68, 65, ...], "A2": [...]}
        self._history: dict[str, list[int]] = {}

    def compute_engagement(self, seat: str, emotion: str, confidence: float) -> tuple[int, str]:
        """
        Hitung engagement_score & status untuk 1 seat, berdasarkan
        emotion+confidence saat ini DAN histori beberapa frame terakhir.
        """
        raw_score = self._emotion_to_score(emotion, confidence)
        self._update_history(seat, raw_score)

        smoothed_score = self._smoothed_score(seat)
        status = self._determine_status(seat, smoothed_score)

        return smoothed_score, status

    def _emotion_to_score(self, emotion: str, confidence: float) -> int:
        """
        Base weight emosi di-skalakan oleh confidence.
        Kalau confidence rendah (model kurang yakin), skornya ditarik
        mendekati netral (50) supaya tidak terlalu ekstrem dari 1 prediksi
        yang tidak meyakinkan.
        """
        base_weight = EMOTION_WEIGHTS.get(emotion, DEFAULT_WEIGHT_IF_UNKNOWN)
        neutral_point = 50
        adjusted = neutral_point + (base_weight - neutral_point) * confidence
        return round(adjusted)

    def _update_history(self, seat: str, score: int) -> None:
        history = self._history.setdefault(seat, [])
        history.append(score)
        # Jaga histori maksimal sesuai temporal_window_size, buang yang lama
        if len(history) > settings.temporal_window_size:
            history.pop(0)

    def _smoothed_score(self, seat: str) -> int:
        """Rolling average dari histori, biar 1 frame aneh gak langsung bikin lonjakan."""
        history = self._history[seat]
        return round(sum(history) / len(history))

    def _determine_status(self, seat: str, smoothed_score: int) -> str:
        """
        Status "Need Attention" HANYA muncul kalau engagement rendah
        secara KONSISTEN (beberapa frame berturut-turut), bukan dari
        1 frame doang — supaya tidak false alarm dari gerakan sesaat
        (misal siswa nunduk ambil pulpen jatuh).
        """
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
        """Reset histori 1 seat — berguna kalau mau restart sesi/kelas baru."""
        self._history.pop(seat, None)