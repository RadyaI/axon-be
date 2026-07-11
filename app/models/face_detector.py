"""
Wrapper untuk MediaPipe Face Detector (Tasks API).

Kenapa pakai Tasks API (bukan mp.solutions yang lama):
MediaPipe versi 0.10.31+ sudah menghapus API lama `mp.solutions.face_detection`.
Tasks API adalah cara resmi yang didukung di versi terbaru.

Class ini HANYA bertugas mendeteksi & crop wajah. Tidak melakukan
identifikasi/pengenalan identitas siapa pun — sesuai prinsip privasi
yang sudah didiskusikan dengan dosen pembimbing.
"""

from pathlib import Path

import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Path ke model .tflite yang sudah didownload manual (lihat Step 3a).
MODEL_PATH = Path(__file__).parent / "weights" / "blaze_face_short_range.tflite"


class FaceDetectorService:
    """
    Wrapper untuk deteksi & crop wajah dari 1 foto.

    Menggunakan running_mode=IMAGE karena tiap request dari ESP32-CAM
    adalah 1 foto berdiri sendiri (bukan stream video kontinu), jadi
    tidak butuh mode VIDEO/LIVE_STREAM yang lebih kompleks.
    """

    def __init__(self, min_detection_confidence: float = 0.5, margin_ratio: float = 0.2) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model face detector tidak ditemukan di {MODEL_PATH}. "
                "Download dulu sesuai instruksi Step 3a."
            )

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        )
        self._detector = FaceDetector.create_from_options(options)
        # Margin tambahan di sekeliling bounding box wajah, supaya crop
        # tidak terlalu ketat mepet wajah (EmotiEffLib butuh sedikit
        # konteks di sekitar wajah untuk akurasi lebih baik).
        self._margin_ratio = margin_ratio

    def detect_and_crop(self, image_rgb: np.ndarray) -> np.ndarray | None:
        """
        Deteksi wajah dari 1 foto (format RGB, HWC, uint8).

        Return:
            Cropped face image (RGB, uint8) jika wajah terdeteksi,
            None jika tidak ada wajah sama sekali.

        Catatan: kalau ada lebih dari 1 wajah kedetek dalam 1 foto
        (misal ada orang lewat di belakang), kita ambil yang PALING
        BESAR bounding box-nya — asumsinya itu wajah siswa yang duduk
        di depan kamera, bukan orang di background.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._detector.detect(mp_image)

        if not result.detections:
            return None

        largest_detection = max(
            result.detections,
            key=lambda d: d.bounding_box.width * d.bounding_box.height,
        )

        return self._crop_with_margin(image_rgb, largest_detection.bounding_box)

    def _crop_with_margin(self, image_rgb: np.ndarray, bbox) -> np.ndarray:
        img_h, img_w = image_rgb.shape[:2]

        margin_x = int(bbox.width * self._margin_ratio)
        margin_y = int(bbox.height * self._margin_ratio)

        x1 = max(0, bbox.origin_x - margin_x)
        y1 = max(0, bbox.origin_y - margin_y)
        x2 = min(img_w, bbox.origin_x + bbox.width + margin_x)
        y2 = min(img_h, bbox.origin_y + bbox.height + margin_y)

        return image_rgb[y1:y2, x1:x2]