"""
Wrapper untuk MediaPipe Face Landmarker (Tasks API).

Menghasilkan 3 hal dari 1 foto: crop wajah, EAR (deteksi mata tertutup),
dan yaw (deteksi kepala menengok) -- ketiganya dari 1 model landmark yang
sama, tidak perlu model tambahan.
"""

import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = Path(__file__).parent / "weights" / "face_landmarker.task"

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Index landmark buat head pose (solvePnP): ujung hidung, dagu, sudut
# mata kiri/kanan, sudut mulut kiri/kanan -- titik yang stabil & gampang
# dikorelasikan ke model wajah 3D generik.
POSE_LANDMARK_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 263,
    "right_eye_corner": 33,
    "left_mouth_corner": 291,
    "right_mouth_corner": 61,
}

# Titik acuan wajah 3D generik dalam satuan mm (model rata-rata manusia,
# BUKAN diambil dari wajah spesifik siapa pun -- jadi tidak melanggar
# prinsip privasi proyek ini).
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),          # nose_tip
    (0.0, -330.0, -65.0),     # chin
    (-225.0, 170.0, -135.0),  # left_eye_corner
    (225.0, 170.0, -135.0),   # right_eye_corner
    (-150.0, -150.0, -125.0), # left_mouth_corner
    (150.0, -150.0, -125.0),  # right_mouth_corner
], dtype=np.float64)


class FaceDetectorService:
    def __init__(self, max_faces: int = 3, margin_ratio: float = 0.2) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model face landmarker tidak ditemukan di {MODEL_PATH}."
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=max_faces,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._margin_ratio = margin_ratio

    def detect(self, image_rgb: np.ndarray) -> tuple[np.ndarray, float, float] | None:
        """
        Return: (face_crop, ear, yaw_degrees) jika wajah terdeteksi,
        None jika tidak ada wajah sama sekali.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        img_h, img_w = image_rgb.shape[:2]

        largest_landmarks = max(
            result.face_landmarks,
            key=lambda lm: self._bbox_area(lm, img_w, img_h),
        )

        face_crop = self._crop_with_margin(image_rgb, largest_landmarks, img_w, img_h)
        ear = self._compute_ear(largest_landmarks, img_w, img_h)
        yaw = self._estimate_yaw(largest_landmarks, img_w, img_h)

        return face_crop, ear, yaw

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
        left_ear = self._eye_ratio(landmarks, LEFT_EYE_IDX, img_w, img_h)
        right_ear = self._eye_ratio(landmarks, RIGHT_EYE_IDX, img_w, img_h)
        return (left_ear + right_ear) / 2.0

    def _eye_ratio(self, landmarks, eye_idx: list[int], img_w: int, img_h: int) -> float:
        pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_idx]

        def dist(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        vertical = dist(pts[1], pts[5]) + dist(pts[2], pts[4])
        horizontal = dist(pts[0], pts[3])

        if horizontal == 0:
            return 0.0
        return vertical / (2.0 * horizontal)

    def _estimate_yaw(self, landmarks, img_w: int, img_h: int) -> float:
        """
        Estimasi sudut yaw (nengok kiri/kanan) dalam derajat, pakai
        cv2.solvePnP: membandingkan titik 2D di foto dengan model
        wajah 3D generik, lalu menghitung rotasi kamera relatif terhadap
        wajah tersebut.

        Kalau solvePnP gagal (kadang terjadi kalau landmark kurang
        stabil), return 0.0 (dianggap "menghadap depan") -- lebih aman
        daripada melempar exception yang bikin seluruh request gagal.
        """
        points_2d = np.array([
            (
                landmarks[POSE_LANDMARK_IDX[name]].x * img_w,
                landmarks[POSE_LANDMARK_IDX[name]].y * img_h,
            )
            for name in POSE_LANDMARK_IDX
        ], dtype=np.float64)

        camera_matrix = np.array([
            [img_w, 0, img_w / 2],
            [0, img_w, img_h / 2],
            [0, 0, 1],
        ], dtype=np.float64)

        success, rotation_vec, _ = cv2.solvePnP(
            FACE_3D_MODEL, points_2d, camera_matrix, None
        )

        if not success:
            return 0.0

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        sy = math.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)

        yaw_rad = math.atan2(-rotation_mat[2, 0], sy)
        return math.degrees(yaw_rad)