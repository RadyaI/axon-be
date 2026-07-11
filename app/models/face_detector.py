"""
Wrapper untuk MediaPipe Face Landmarker (Tasks API).

Kenapa ganti dari FaceDetector ke FaceLandmarker:
FaceDetector cuma ngasih bounding box wajah, gak ngasih titik detail
kayak mata/mulut. Padahal kita butuh landmark mata buat hitung Eye
Aspect Ratio (EAR) -- sinyal buat deteksi mata tertutup (ngantuk/tidur),
kasus yang sebelumnya sering salah kebaca sebagai "Sadness" oleh model
emosi. FaceLandmarker sekaligus mendeteksi wajah DAN landmark, jadi
gantiin 2 kebutuhan (deteksi + landmark) dengan 1 model.

Class ini HANYA mendeteksi & crop wajah + landmark. Tidak melakukan
identifikasi/pengenalan identitas siapa pun -- sesuai prinsip privasi
yang sudah didiskusikan dengan dosen pembimbing.
"""

import math
from pathlib import Path

import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Path ke model .task yang sudah didownload manual dari MediaPipe model zoo.
MODEL_PATH = Path(__file__).parent / "weights" / "face_landmarker.task"

# --- Index landmark mata (dari 478 titik face mesh MediaPipe) ---
# Urutan: [kiri, atas1, atas2, kanan, bawah1, bawah2] -- dipakai buat EAR.
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


class FaceDetectorService:
    """
    Wrapper untuk deteksi wajah + landmark dari 1 foto.

    Menggunakan running_mode=IMAGE karena tiap request dari ESP32-CAM
    adalah 1 foto berdiri sendiri (bukan stream video kontinu).
    """

    def __init__(self, max_faces: int = 3, margin_ratio: float = 0.2) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model face landmarker tidak ditemukan di {MODEL_PATH}. "
                "Download dulu file face_landmarker.task dari MediaPipe model zoo."
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=max_faces,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._margin_ratio = margin_ratio

    def detect(self, image_rgb: np.ndarray) -> tuple[np.ndarray, float] | None:
        """
        Deteksi wajah dari 1 foto (format RGB, HWC, uint8), lalu hitung EAR-nya.

        Return:
            (face_crop, ear) jika wajah terdeteksi, None jika tidak ada wajah.
            Kalau ada lebih dari 1 wajah, ambil yang bounding box-nya PALING
            BESAR -- asumsinya itu siswa yang duduk di depan kamera, bukan
            orang lewat di background.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        img_h, img_w = image_rgb.shape[:2]

        # Pilih wajah dengan bounding box terbesar (dihitung dari landmark,
        # karena FaceLandmarker gak ngasih bbox langsung kayak FaceDetector).
        largest_landmarks = max(
            result.face_landmarks,
            key=lambda lm: self._bbox_area(lm, img_w, img_h),
        )

        face_crop = self._crop_with_margin(image_rgb, largest_landmarks, img_w, img_h)
        ear = self._compute_ear(largest_landmarks, img_w, img_h)

        return face_crop, ear

    def _bbox_area(self, landmarks, img_w: int, img_h: int) -> float:
        xs = [lm.x * img_w for lm in landmarks]
        ys = [lm.y * img_h for lm in landmarks]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    def _crop_with_margin(self, image_rgb: np.ndarray, landmarks, img_w: int, img_h: int) -> np.ndarray:
        xs = [lm.x * img_w for lm in landmarks]
        ys = [lm.y * img_h for lm in landmarks]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        margin_x = (x_max - x_min) * self._margin_ratio
        margin_y = (y_max - y_min) * self._margin_ratio

        x1 = max(0, int(x_min - margin_x))
        y1 = max(0, int(y_min - margin_y))
        x2 = min(img_w, int(x_max + margin_x))
        y2 = min(img_h, int(y_max + margin_y))

        return image_rgb[y1:y2, x1:x2]

    def _compute_ear(self, landmarks, img_w: int, img_h: int) -> float:
        """
        Eye Aspect Ratio -- rasio tinggi:lebar bukaan mata.
        Mata terbuka normal ~0.25-0.35, mata tertutup mendekati 0.

        Dihitung dari rata-rata mata kiri & kanan, biar lebih stabil
        kalau 1 sisi wajah agak miring dari kamera.
        """
        left_ear = self._eye_ratio(landmarks, LEFT_EYE_IDX, img_w, img_h)
        right_ear = self._eye_ratio(landmarks, RIGHT_EYE_IDX, img_w, img_h)
        return (left_ear + right_ear) / 2.0

    def _eye_ratio(self, landmarks, eye_idx: list[int], img_w: int, img_h: int) -> float:
        # Landmark MediaPipe itu normalized (0.0-1.0), WAJIB dikaliin
        # dimensi asli gambar dulu sebelum hitung jarak Euclidean --
        # kalau enggak, hasilnya distorsi buat foto yang gak persegi.
        pts = [
            (landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_idx
        ]

        def dist(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        vertical = dist(pts[1], pts[5]) + dist(pts[2], pts[4])
        horizontal = dist(pts[0], pts[3])

        if horizontal == 0:
            return 0.0
        return vertical / (2.0 * horizontal)