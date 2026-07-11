"""
Endpoint HTTP untuk menerima foto dari ESP32-CAM.

File ini sengaja TIPIS (thin layer) — hanya menangani validasi HTTP
request/response, semua logic sesungguhnya ada di app/services/.
"""

from fastapi import APIRouter, Form, HTTPException, UploadFile

from app.schemas.prediction import PredictionResponse
from app.services.engagement_service import process_frame

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile,
    seat: str = Form(...),
) -> PredictionResponse:
    """
    Terima 1 foto dari ESP32-CAM + seat_id, kembalikan hasil analisis
    engagement.

    Foto HANYA diproses di memori (RAM) melalui `image.read()` di bawah —
    tidak pernah ditulis ke disk, sesuai prinsip privasi proyek ini.
    """
    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="File gambar kosong.")

    try:
        result = await process_frame(image_bytes, seat)
    except ValueError as e:
        # Contoh: seat tidak valid (ditangani oleh Pydantic validator)
        raise HTTPException(status_code=422, detail=str(e))

    return result