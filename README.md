---
title: Stroke Detect Pro API
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
pinned: false
---

# Stroke Detect Pro API 

<div align="center">

**Sistem Deteksi Multimodal Berbasis AI untuk Indikasi Stroke**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.32-orange.svg)](https://mediapipe.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg)](https://scikit-learn.org/)

</div>

## 1. Deskripsi

**Stroke Detect Pro API** adalah REST API berkinerja tinggi yang dibangun menggunakan **FastAPI** untuk mendeteksi tanda-tanda awal indikasi Stroke. API ini menggunakan pendekatan multimodal (3 metode analisis) untuk memberikan hasil yang komprehensif:

1. **Vision Analysis (MediaPipe & OpenCV)**: Deteksi asimetri wajah (Facial Palsy) dan koordinasi arah lirikan mata (Eye Symmetry).
2. **Tabular Prediction (Scikit-Learn Random Forest)**: Analisis persentase risiko stroke berdasarkan metrik data medis pasien yang dikenal sebagai *Riskometer*.
3. **Speech Analysis (PyTorch ResNet-18)**: Analisis rekaman suara (Mel-Spectrogram) untuk klasifikasi *binary* deteksi penderita *Dysarthria* (bicara cadel/tidak jelas).

### Teknologi yang Digunakan

| Komponen | Teknologi | Fungsi |
|----------|-----------|--------|
| **Framework Web** | FastAPI, Pydantic | REST API, Validasi Data, Swagger UI |
| **Vision AI** | MediaPipe, OpenCV | Deteksi landmark wajah & kalkulasi asimetri |
| **Tabular AI** | Scikit-Learn, Pandas | Prediksi *machine learning* berbasis data medis |
| **Audio AI** | PyTorch, Torchaudio | Ekstraksi fitur audio & klasifikasi *Deep Learning* |
| **Deployment** | Docker, Hugging Face | Hosting & manajemen container yang dioptimasi |

---

## 2. Struktur Direktori & Arsitektur

```text
stroke-detect-pro/
├── app/
│   ├── main.py                    # Entry point FastAPI application
│   ├── config.py                  # Environment setup & konfigurasi batas file
│   ├── schemas.py                 # Pydantic models untuk request body (Swagger)
│   ├── api/
│   │   └── routes.py              # Seluruh HTTP Endpoints
│   └── services/
│       ├── speech_service.py      # Logika PyTorch Audio (Dysarthria)
│       ├── riskometer_service.py  # Logika Scikit-Learn (Stroke Riskometer)
│       └── facial_service.py      # Logika MediaPipe & OpenCV (Facial Palsy & Gaze)
├── model/                         # Folder Git LFS
│   ├── face_landmarker.task                  # Model MediaPipe Vision
│   ├── model_scripted_resnet18_cpu-binary.pt # Model PyTorch Audio
│   ├── random_forest_model.joblib            # Model Prediksi Tabular
│   └── scaler.joblib                         # Scaler Data Medis Tabular
├── .env                           # Environment variables (API_KEY, dll)
├── .gitattributes                 # Konfigurasi Git LFS
├── Dockerfile                     # Setup Container Production / HF Spaces
├── requirements.txt               # Daftar dependensi Python
└── README.md                      # Dokumentasi Proyek
```

### Penjelasan File Penting

| File | Fungsi |
|------|--------|
| `app/main.py` | Inisialisasi FastAPI, setup metadata dokumentasi Swagger, middleware CORS. |
| `app/config.py` | Load env vars, API Key validation (`X-API-Key`), MIME types, dan batas file (5MB). |
| `app/api/routes.py` | Rute *endpoint* API, serta *helper* untuk konversi citra ke JPEG Base64 yang ramah Mobile. |
| `app/services/*` | Memisahkan logika AI berbasis *Service Pattern* (Vision, Riskometer, Speech). |
| `model/` | Penyimpanan pra-terlatih (*pre-trained*) menggunakan **Git LFS** karena ukurannya yang besar. |

---

## 3. Instalasi & Setup (Development)

### Prerequisites
- Python 3.10+
- Git & Git LFS terinstal di sistem.

### Step 1: Clone Repository & Setup Git LFS
Karena ada file model AI yang besar (*joblib, .pt, .task*), **wajib** menggunakan Git LFS.
```bash
git clone <repository-url>
cd stroke-detect-pro
git lfs install
git lfs pull
```

### Step 2: Virtual Environment & Dependencies
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Konfigurasi Environment (`.env`)
Buat file `.env` di direktori utama.
```env
API_KEY=your_secret_api_key_here
ALLOWED_ORIGINS=*
MAX_FILE_SIZE_MB=5
```

### Step 4: Jalankan Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Buka browser dan akses antarmuka Swagger UI di: **`http://localhost:8000/docs`**

---

## 4. Dokumentasi Endpoint API

*Catatan Penting: Semua endpoint wajib menyertakan API Key pada Header:*  
`X-API-Key: <your_secret_api_key_here>`

### 1️⃣ VISION: Deteksi Facial Palsy (Asimetri Mulut & Wajah)
Mendeteksi asimetri otot wajah yang menjadi indikator kuat stroke.

- **URL:** `POST /api/v1/analyze/facial-palsy`
- **Request:** `multipart/form-data` -> `file` (Gambar JPG/PNG)
- **Response:**
```json
{
  "status": "success",
  "analysis": {
    "severity_score": 85,
    "status_label": "Asimetri Parah",
    "metrics": {
      "raw_severity_pct": 85,
      "mouth_diff_deg": 12.5,
      "eye_asymmetry_ratio": 0.15
    }
  },
  "image_result": "data:image/jpeg;base64,/9j/4AAQSk..."
}
```

### 2️⃣ VISION: Deteksi Asimetri Mata (Gaze Symmetry)
Mengevaluasi keselarasan koordinasi arah lirikan pupil mata menggunakan rasio *Medial-Lateral*.

- **URL:** `POST /api/v1/analyze/eye-symmetry`
- **Request:** `multipart/form-data` -> `file` (Gambar JPG/PNG)
- **Response:**
```json
{
  "status": "success",
  "analysis": {
    "severity_score": 60,
    "status_label": "Asimetri Parah",
    "metrics": {
      "gaze_left": 0.5,
      "gaze_right": 0.75,
      "gaze_difference": 0.25
    }
  },
  "image_result": "data:image/jpeg;base64,/9j/4AAQSk..."
}
```

### 3️⃣ TABULAR: Prediksi Riskometer Stroke (Data Medis)
Memprediksi kelas & probabilitas risiko stroke pasien menggunakan algoritma *Random Forest*.

- **URL:** `POST /api/v1/predict/riskometer`
- **Request:** `application/json`
```json
{
  "gender": 1,
  "age": 65,
  "hypertension": 1,
  "heart_disease": 0,
  "ever_married": 1,
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "smoking_status": 1
}
```
- **Response:**
```json
{
  "status": "success",
  "data": {
    "severity_score": 79,
    "status_label": "Risiko Tinggi Stroke",
    "metrics": {
      "predicted_class": 1,
      "risk_percentage": 78.5
    }
  }
}
```

### 4️⃣ AUDIO: Deteksi Dysarthria (Analisis Suara)
Menganalisis rekaman suara (*Mel-Spectrogram*) menggunakan model *ResNet-18 Binary* untuk mendeteksi *slurred speech* (Disartria).

- **URL:** `POST /api/v1/analyze/speech-dysarthria`
- **Request:** `multipart/form-data` -> `file` (Audio berektensi `.wav`)
- **Response:**
```json
{
  "status": "success",
  "filename": "pasien_audio_01.wav",
  "data": {
    "severity_score": 92,
    "status_label": "Indikasi Kuat Disartria",
    "metrics": {
      "predicted_class": "Dysarthria",
      "confidence_percentage": 92.4,
      "probabilities": {
        "Dysarthria": 92.4,
        "Non-Dysarthria": 7.6
      }
    }
  }
}
```

---

## 5. Deployment & Production (Docker)

Proyek ini menggunakan **Docker** dan **Gunicorn** dengan Uvicorn *workers*, yang telah dioptimasi secara spesifik untuk memproses beban kerja *Machine Learning* tinggi.

### Build & Run Docker Lokal

1. **Build Image Docker**
```bash
docker build -t stroke-detect-api .
```

2. **Jalankan Container**
```bash
docker run -d -p 8000:8000 --env-file .env --name stroke-api stroke-detect-api
```
Server akan berjalan di `http://localhost:8000`.

### Standar Industri Dockerfile Proyek Ini
Aplikasi ini sudah dipersiapkan sebagai konfigurasi *Production-Ready*, terutama jika di-*deploy* ke Hugging Face Spaces:
- **PyTorch CPU-Only Efficiency:** Mengunduh modul AI versi `cpu` (`--extra-index-url https://download.pytorch.org/whl/cpu`) memangkas ukuran *image* Docker lebih dari 2GB (karena tidak mengunduh library CUDA GPU).
- **Sistem Dependensi C & C++ yang Utuh:** Sudah dilengkapi pustaka mesin Linux `libgl1` (mutlak dibutuhkan OpenCV untuk pemrosesan citra) dan `libsndfile1` (dibutuhkan `soundfile` untuk pemrosesan audio `.wav`).
- **Antisipasi Timeout (*Gunicorn*):** Timeout *worker* dipasang di angka `120 detik` (`--timeout 120`). Karena *model inference* (seperti ResNet18 dan MediaPipe) terkadang membutuhkan durasi komputasi yang bervariasi, konfigurasi ini mencegah gangguan koneksi API (seperti error HTTP 504).

---

## 6. Changelog

### v1.0.0 (Current Release)
- **[Fitur Baru]** Integrasi Tabular Random Forest Model (Riskometer dengan persentase skor risiko).
- **[Fitur Baru]** Integrasi PyTorch Audio ResNet18 Binary class (Dysarthria vs Non-Dysarthria).
- **[Peningkatan]** Refactor arsitektur menjadi *Services Pattern* (`facial_service.py`, `riskometer_service.py`, `speech_service.py`).
- **[Peningkatan]** *Helper* enkripsi *Base64* otomatis melakuan rasio kompresi gambar (lebar maks 720px) agar respons API ringan saat di-parsing oleh perangkat Mobile/Frontend.
- **[Peningkatan]** Standardisasi output response (`severity_score`, `status_label`, `metrics`) pada semua endpoints AI.
- **[Peningkatan]** Penambahan Pydantic Schemas untuk membangun OpenAPI/Swagger UI yang rapi.

---

<div align="center">
<b>© 2026 Stroke Detect Pro API Team</b>
</div>