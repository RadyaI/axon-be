"""
Wrapper untuk EmotiEffLib — emotion recognition dari crop wajah.

Class ini HANYA mengembalikan emotion + confidence mentah.
engagement_score DIHITUNG TERPISAH oleh Rule Engine (app/core/rule_engine.py,
dibuat di step berikutnya) — sesuai desain awal supaya logic AI (emosi apa
yang terdeteksi) terpisah dari logic bisnis (bagaimana kita menginterpretasi
emosi itu jadi skor engagement).
"""

import numpy as np
from emotiefflib.facial_analysis import EmotiEffLibRecognizer

from app.core.config import settings


class EmotionRecognizerService:
    """
    Wrapper untuk prediksi emosi dari 1 crop wajah menggunakan EmotiEffLib.

    engine="onnx" dipilih (bukan "torch") supaya inference ringan di CPU
    laptop biasa tanpa GPU, sesuai requirement awal proyek.
    """

    def __init__(self) -> None:
        self._recognizer = EmotiEffLibRecognizer(
            engine="onnx",
            model_name=settings.emotiefflib_model_name,
            device="cpu",
        )

    def predict(self, face_crop_rgb: np.ndarray) -> tuple[str, float]:
        """
        Prediksi emosi dari 1 crop wajah (RGB, HWC, uint8).

        Return:
            (emotion_label, confidence) — confidence dihitung dari softmax
            atas raw logits, supaya nilainya selalu di rentang 0.0-1.0
            (logits mentah dari model bisa berupa angka apa saja, positif
            atau negatif, jadi tidak representatif sebagai "confidence").
        """
        # predict_emotions menerima LIST gambar (batch), meski kita cuma
        # punya 1 wajah — jadi dibungkus list, lalu ambil elemen [0] hasilnya.
        emotions, scores = self._recognizer.predict_emotions(
            [face_crop_rgb], logits=True
        )

        emotion_label = emotions[0]
        confidence = self._softmax_confidence(scores[0])

        return emotion_label, confidence

    @staticmethod
    def _softmax_confidence(logits: np.ndarray) -> float:
        """Konversi raw logits jadi probabilitas (softmax), ambil nilai max-nya."""
        exp_logits = np.exp(logits - np.max(logits))  # dikurangi max untuk stabilitas numerik
        probabilities = exp_logits / exp_logits.sum()
        return float(np.max(probabilities))