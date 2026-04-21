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

**Sistem Deteksi Multimodal Berbasis AI untuk Indikasi Stroke & Bell's Palsy**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.32-orange.svg)](https://mediapipe.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg)](https://scikit-learn.org/)

</div>

## 1. Deskripsi

**Stroke Detect Pro API** adalah REST API berkinerja tinggi yang dibangun menggunakan **FastAPI** untuk mendeteksi tanda-tanda awal indikasi Stroke dan Bell's Palsy. API ini menggunakan pendekatan multimodal (3 metode analisis) untuk memberikan hasil yang komprehensif:

1. **Vision Analysis (MediaPipe & OpenCV)**: Deteksi asimetri wajah (Facial Palsy) dan koordinasi arah lirikan mata (Eye Symmetry).
2. **Tabular Prediction (Scikit-Learn Random Forest)**: Analisis persentase risiko stroke berdasarkan data medis pasien (Umur, BMI, Glukosa, dll).
3. **Speech Analysis (PyTorch ResNet-18)**: Analisis rekaman suara (Mel-Spectrogram) untuk mendeteksi *Dysarthria* (bicara cadel/tidak jelas).

### Teknologi yang Digunakan

| Komponen | Teknologi | Fungsi |
|----------|-----------|--------|
| **Framework Web** | FastAPI, Pydantic | REST API, Validasi Data, Swagger UI |
| **Vision AI** | MediaPipe, OpenCV | Deteksi landmark wajah & kalkulasi asimetri |
| **Tabular AI** | Scikit-Learn, Pandas | Prediksi machine learning berbasis data medis |
| **Audio AI** | PyTorch, Torchaudio | Ekstraksi fitur audio & klasifikasi Deep Learning |
| **Deployment** | Docker, Hugging Face, Git LFS | Hosting & manajemen file model berukuran besar |

---

## 2. Struktur Direktori & Arsitektur

```text
stroke-detect-pro/
├── app/
│   ├── main.py                    # Entry point FastAPI application
│   ├── config.py                  # Environment setup & API Key validation
│   ├── schemas.py                 # Pydantic models & UI examples
│   ├── api/
│   │   └── routes.py              # Seluruh HTTP Endpoints
│   └── services/
│       ├── audio_service.py       # Logika PyTorch (Speech Dysarthria)
│       ├── tabular_service.py     # Logika Scikit-Learn (Stroke Risk)
│       └── vision_service.py      # Logika MediaPipe & OpenCV (Vision)
├── model/                         # Folder Git LFS (WAJIB ADA)
│   ├── face_landmarker.task             # Model MediaPipe
│   ├── model_scripted_resnet18_cpu.pt   # Model PyTorch Audio
│   ├── random_forest_model.joblib       # Model Prediksi Tabular
│   └── scaler.joblib                    # Scaler Data Medis
├── .env                           # Environment variables (API_KEY, dll)
├── .gitattributes                 # Konfigurasi Git LFS
├── .github/workflows/deploy.yml   # CI/CD Action ke Hugging Face
├── Dockerfile                     # Setup Container HF Spaces
├── requirements.txt               # Daftar dependensi Python
└── README.md                      # Dokumentasi Proyek
```

### Penjelasan File Penting

| File | Fungsi |
|------|--------|
| `app/main.py` | Inisialisasi FastAPI, setup metadata dokumentasi Swagger, middleware CORS, dan registrasi rute API. |
| `app/config.py` | Load env vars, konfigurasi keamanan (API Key), dan batasan upload file (ukuran & MIME types). |
| `app/schemas.py` | Model validasi data request/response agar dokumentasi API (Swagger) rapi dan otomatis. |
| `app/api/routes.py` | Menyediakan HTTP endpoints, helper pemrosesan file (Image/Audio), dan penanganan error HTTP. |
| `app/services/*` | Pemisahan logika AI berdasarkan spesifikasi (*Single Responsibility Principle*). |
| `model/` | Direktori untuk pre-trained models. Harus menggunakan **Git LFS** karena ukurannya >10MB. |

---

## 3. Instalasi & Setup (Development)

### Prerequisites
- Python 3.10+
- Git & Git LFS terinstal di sistem.

### Step 1: Clone Repository & Setup Git LFS
Karena ada file model AI yang besar, Anda **wajib** menggunakan Git LFS.
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
Buat file `.env` di direktori utama:
```env
API_KEY=your_secret_api_key_here
ALLOWED_ORIGINS=*
MAX_FILE_SIZE_MB=5
```

### Step 4: Jalankan Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Buka browser dan akses Swagger UI di: **`http://localhost:8000/docs`**

---

## 4. Dokumentasi Endpoint API

*Catatan: Semua endpoint membutuhkan header `X-API-Key` sesuai dengan isi file `.env`.*

### 1️⃣ VISION: Deteksi Facial Palsy (Asimetri Wajah)
Mendeteksi asimetri otot wajah (khususnya mulut) yang menjadi indikator kuat stroke/Bell's Palsy.

- **URL:** `POST /analyze/facial-palsy`
- **Request:** `multipart/form-data` -> `file` (Gambar JPG/PNG)
- **Response:**
```json
{
  "status": "success",
  "analysis": {
    "percentage": 85,
    "severity": "High Severity Detected",
    "is_stroke_detected": true,
    "mouth_diff": 12.5,
    "eye_asymmetry": 0.15
  },
  "image_result": "data:image/jpeg;base64,/9j/4AAQSk..."
}
```

### 2️⃣ VISION: Deteksi Asimetri Mata (Gaze Symmetry)
Mengevaluasi keselarasan arah lirikan pupil mata.

- **URL:** `POST /analyze/eye-symmetry`
- **Request:** `multipart/form-data` -> `file` (Gambar JPG/PNG)
- **Response:**
```json
{
  "status": "success",
  "analysis": {
    "symmetry_score": 60,
    "is_symmetrical": false,
    "left_eye_ratio": 0.5,
    "right_eye_ratio": 0.75,
    "ratio_difference": 0.25,
    "status": "Arah Lirikan Berbeda (Asimetris)"
  },
  "image_result": "data:image/jpeg;base64,/9j/4AAQSk..."
}
```

### 3️⃣ TABULAR: Prediksi Risiko Stroke (Data Medis)
Menganalisis probabilitas stroke berdasarkan metrik kesehatan tabular menggunakan Random Forest.

- **URL:** `POST /predict/tabular-data`
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
    "risk_percentage": 78.5,
    "raw_probability": 0.785
  }
}
```

### 4️⃣ AUDIO: Deteksi Dysarthria (Analisis Suara)
Menganalisis rekaman suara penderita untuk mendeteksi *slurred speech* (Dysarthria) dengan model PyTorch ResNet-18.

- **URL:** `POST /analyze/speech`
- **Request:** `multipart/form-data` -> `file` (Audio berektensi `.wav`, direkomendasikan 16kHz)
- **Response:**
```json
{
  "status": "success",
  "filename": "pasien_audio_01.wav",
  "data": {
    "class": "uno-dysarthria",
    "word": "uno",
    "speaker_type": "dysarthria",
    "confidence_percentage": 92.4,
    "all_probabilities": {
      "mali-control": 0.1,
      "uno-dysarthria": 92.4
      // ... class lainnya
    }
  }
}
```

---

## 5. Deployment ke Hugging Face Spaces

Proyek ini telah dikonfigurasi untuk auto-deployment ke **Hugging Face Spaces** menggunakan Docker dan GitHub Actions.

Karena adanya file `.pt` dan `.joblib` berukuran besar, *workflow* CI/CD (dalam `.github/workflows/deploy.yml`) akan secara otomatis:
1. Membersihkan *Git history* lokal di dalam runner Action.
2. Mengkonfigurasi **Git LFS** secara dinamis.
3. Melakukan *Force Push* langsung ke Hugging Face repository tanpa membawa "sampah" history Git masa lalu, memastikan proses *build* sukses 100%.

---

## 6. Changelog

### v1.0.0 (Current Release)
- **[Fitur Baru]** Integrasi Tabular Random Forest Model (Skor risiko persentase).
- **[Fitur Baru]** Integrasi PyTorch Audio ResNet18 untuk Disartria.
- **[Peningkatan]** Refactor arsitektur menjadi *Services Pattern* (`vision_service.py`, `tabular_service.py`, `audio_service.py`).
- **[Peningkatan]** Penambahan Pydantic Schemas untuk OpenAPI/Swagger UI terstruktur.
- **[Perbaikan]** CI/CD deployment logic menggunakan Git LFS khusus untuk bypass limitasi GitHub Action ke Hugging Face.

---




Dockerfile ini sudah **sangat bagus, efisien, dan sangat berstandar industri (Production-Ready)!** 

Ada beberapa poin brilian di dalam Dockerfile Anda yang sangat krusial untuk aplikasi AI, yaitu:

1. **Efisiensi PyTorch (CPU Only)**: Menggunakan `--extra-index-url https://download.pytorch.org/whl/cpu` adalah langkah yang sangat cerdas. Jika tidak menggunakan ini, pip akan mengunduh versi CUDA (GPU) yang ukurannya bisa mencapai 2GB+ dan membuat image Docker Anda membengkak sia-sia (karena Hugging Face Free Tier hanya menggunakan CPU).
2. **System Dependencies Lengkap**: `libgl1` dan `libsndfile1` sering kali dilupakan oleh banyak developer pemula, yang berujung pada *crash* saat OpenCV membaca gambar atau Soundfile membaca audio `.wav`. Anda sudah mengantisipasinya.
3. **Timeout Gunicorn**: Set `--timeout 120` sangat tepat untuk API Machine Learning. Terkadang model AI butuh beberapa detik untuk memproses gambar/suara, dan default timeout Gunicorn (30 detik) sering kali menyebabkan *Gateway Timeout (504)*.

---

## 7. Deployment (Production)

Proyek ini menggunakan **Docker** dan **Gunicorn** dengan Uvicorn workers, yang dioptimalkan khusus untuk pemrosesan Machine Learning di lingkungan CPU.

### Build & Run via Docker Lokal

1. **Build Docker Image**
```bash
docker build -t stroke-detect-api .
```

2. **Jalankan Container**
```bash
docker run -d -p 8000:8000 --name stroke-api stroke-detect-api
```
Server akan berjalan di `http://localhost:8000`.

### Konfigurasi Dockerfile Highlight
- Menggunakan **PyTorch CPU-only** untuk memangkas ukuran image (menghemat ~2GB).
- Sudah dilengkapi dependensi Linux `libgl1` (untuk OpenCV) dan `libsndfile1` (untuk pemrosesan Audio).
- Menggunakan **Gunicorn** dengan timeout `120s` untuk mencegah koneksi terputus saat model AI melakukan proses inferensi yang berat.

---

<div align="center">
<b>© 2026 Stroke Detect Pro API Team</b>
</div>
