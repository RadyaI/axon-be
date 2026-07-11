# AXON Backend

Backend AI untuk sistem monitoring engagement siswa berbasis **FastAPI**, **MediaPipe**, **EmotiEffLib**, dan **Firebase**.

## Tech Stack

* Python 3.12
* FastAPI
* OpenCV
* MediaPipe
* EmotiEffLib
* ONNX Runtime
* Firebase Admin SDK

---

# Persiapan

## 1. Clone Repository

```bash
git clone <URL_REPOSITORY>
cd axon
```

---

## 2. Install Python

Disarankan menggunakan **Python 3.12.x**.

Cek versi:

```bash
py --version
```

---

## 3. Buat Virtual Environment

Windows

```bash
py -3.12 -m venv venv
```

Aktifkan

PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

CMD

```cmd
venv\Scripts\activate.bat
```

---

## 4. Install Dependency

```bash
pip install -r requirements.txt
```

---

# Konfigurasi

## 1. Firebase

Karena file Firebase tidak disimpan di repository, setiap anggota tim harus menambahkan sendiri file:

```text
app/firebase/service_account.json
```

File tersebut dapat diperoleh dari Firebase Console:

```
Project Settings
→ Service Accounts
→ Generate New Private Key
```

---

## 2. Environment Variable

Buat file

```text
.env
```

Contoh:

```env
FIREBASE_COLLECTION=predictions
```

Tambahkan konfigurasi lain sesuai kebutuhan project.

---

## 3. Download Model MediaPipe

Model AI tidak disimpan di GitHub.

Masuk ke folder:

```bash
cd app/models/weights
```

Lalu download model:

```powershell
Invoke-WebRequest `
-Uri "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite" `
-OutFile "blaze_face_short_range.tflite"
```

Setelah selesai struktur folder menjadi:

```text
app/
└── models/
    └── weights/
        └── blaze_face_short_range.tflite
```

> **Catatan:** Jika project nantinya menggunakan MediaPipe Face Landmarker, model `.tflite` ini akan diganti dengan model `.task`.

---

# Menjalankan Backend

Dari root project:

```bash
uvicorn app.main:app --reload
```

Server akan berjalan di:

```
http://127.0.0.1:8000
```

Swagger Documentation:

```
http://127.0.0.1:8000/docs
```
