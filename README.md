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

**Sistem Deteksi Asimetri Wajah Berbasis AI untuk Stroke & Bell's Palsy**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.32-orange.svg)](https://mediapipe.dev/)

</div>

##  1. Deskripsi

REST API berkinerja tinggi yang dibangun menggunakan **FastAPI**, **MediaPipe**, dan **OpenCV** untuk mendeteksi asimetri wajah secara real-time. API ini dirancang untuk membantu **analisis awal indikasi Facial Palsy** (Bell's Palsy, Stroke, dan kondisi neurologis lainnya) dengan hasil yang akurat berupa:

- **Skor Asimetri Wajah** (Persentase)
- **Tingkat Keparahan** (Mild, Moderate, Severe)
- **Deteksi Stroke Awal** (Berdasarkan tanda-tanda klinis)
- **Visualisasi Analisis** (Garis asimetri pada gambar hasil)
- **Metrik Detail** (Perbedaan mata, mulut, dan properti geometri wajah)

### Teknologi yang Digunakan

| Komponen | Teknologi | Fungsi |
|----------|-----------|--------|
| **Framework Web** | FastAPI | REST API berkecepatan tinggi |
| **Face Detection** | MediaPipe Face Landmarker | 468 landmark deteksi wajah presisi tinggi |
| **Image Processing** | OpenCV | Manipulasi & visualisasi gambar |
| **Testing** | Pytest | Unit & integration testing |
| **Dependency Mgmt** | pip + python-dotenv | Package management & konfigurasi |

### Arsitektur Proyek

```
┌─ Modular Architecture
│
├─ app.config       → Setup environment & keamanan API Key
├─ app.services    → Business logic (MediaPipe, OpenCV analysis)
├─ app.api.routes  → HTTP Endpoints & request handling
└─ app.main        → FastAPI app initialization

Keamanan: API Key-based authentication (X-API-Key header)
```

## 2. Struktur Direktori

```text
facial-palsy-api/
├── app/
│   ├── main.py                          # Entry point FastAPI application
│   ├── config.py                        # Environment setup & API Key validation
│   ├── __init__.py
│   ├── api/
│   │   ├── routes.py                    # API endpoints (/analyze/facial-palsy, /analyze/eye-symmetry)
│   │   ├── __init__.py
│   │   └── __pycache__/
│   ├── services/
│   │   ├── ml_service.py                # FaceAnalyzerService class (MediaPipe + OpenCV logic)
│   │   ├── __init__.py
│   │   └── __pycache__/
│   └── __pycache__/
├── model/
│   └── face_landmarker.task             # MediaPipe Face Landmarker Model (auto-downloaded)
├── tests/
│   ├── test_main.py                     # Unit & integration tests
│   ├── __init__.py
│   └── __pycache__/
├── .env                                 # Environment variables (API_KEY, etc.)
├── .gitignore                           # Git ignore rules
├── pytest.ini                           # Pytest configuration
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

### Penjelasan File Penting

| File | Fungsi |
|------|--------|
| `app/main.py` | Inisialisasi FastAPI & middleware CORS |
| `app/config.py` | Load environment variables & API Key validation |
| `app/services/ml_service.py` | Core AI logic: Face detection, landmark computation, asymmetry calculation |
| `app/api/routes.py` | HTTP endpoints dengan input validation & error handling |
| `model/face_landmarker.task` | Pre-trained MediaPipe model (468 face landmarks) |
| `.env` | Sensitive data (API_KEY, ALLOWED_ORIGINS, MAX_FILE_SIZE_MB) |

## 3. Instalasi & Setup

### Prerequisites

Sebelum memulai, pastikan sudah menginstall:

- **Python 3.10 - 3.12** ([Download](https://www.python.org/downloads/))
- **pip** (Python package manager - biasanya included)
- **Git** (optional, untuk version control)

**Verifikasi Python:**
```bash
python --version
pip --version
```

### Step 1: Clone Repository & Navigate ke Folder

```bash
# Jika menggunakan Git
git clone <repository-url>
cd facial-palsy-api

# ATAU simple extract ZIP ke folder
cd path/to/facial-palsy-api
```

### Step 2: Buat Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Jika berhasil, akan ada `(.venv)` di awal terminal Anda.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Ini akan memakan waktu 2-5 menit (terutama MediaPipe dan OpenCV).

**Verifikasi instalasi:**
```bash
pip list
```

### Step 4: Buat File `.env`

Buat file `.env` **di folder root project**:

```bash
# Windows
type nul > .env

# macOS / Linux
touch .env
```

Isi file dengan:

```env
API_KEY=your-super-secret-api-key-here-min-32-chars-recommended
ALLOWED_ORIGINS=http://localhost,http://localhost:3000,http://localhost:8000
MAX_FILE_SIZE_MB=5
```

**Konfigurasi Parameter:**

| Parameter | Nilai Default | Keterangan |
|-----------|---|---|
| `API_KEY` | Required | Gunakan string random yang kuat (min 32 karakter) |
| `ALLOWED_ORIGINS` | `*` | Daftar origin yang boleh akses API (comma-separated) |
| `MAX_FILE_SIZE_MB` | `5` | Ukuran max upload gambar (dalam MB) |

### Step 5: Auto-Download Model AI

Saat server pertama kali dijalankan, model `face_landmarker.task` akan **otomatis di-download** (~35 MB).

## 4. Menjalankan Server

Pastikan virtual environment sudah active, lalu jalankan:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Output yang diharapkan:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete
```

Server sekarang aktif di `http://localhost:8000`

## 5. API Documentation & Penggunaan

### Interactive Docs (Swagger UI)

Buka di browser: **`http://localhost:8000/docs`**

Di sini Anda bisa:
- Lihat semua endpoint
- Test API langsung (try it out)
- Lihat request/response format

## 6. Endpoint Details

### 1️⃣ **POST** `/analyze/facial-palsy`

Mendeteksi asimetri wajah yang menunjukkan indikasi **Facial Palsy** (Bell's Palsy, Stroke).

#### Request

**Headers:**
```http
X-API-Key: your-api-key-from-.env
Content-Type: multipart/form-data
```

**Body (Form-Data):**
- `file` (File, required): Gambar wajah `.jpg`, `.jpeg`, atau `.png`
  - Max size: Sesuai `MAX_FILE_SIZE_MB` di `.env` (default 5 MB)
  - Min resolution: 100x100 pixel
  - Recommended: 480p ke atas untuk akurasi lebih baik

#### Response Success (200 OK)

```json
{
  "status": "success",
  "analysis": {
    "percentage": 32,
    "severity": "Mild Asymmetry",
    "is_stroke_detected": false,
    "mouth_diff": 4.5,
    "eye_asymmetry": 0.09,
    "face_scale": 0.25,
    "asymmetry_score": 18.5
  },
  "image_result": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
}
```

**Response Fields:**

| Field | Type | Keterangan |
|-------|------|-----------|
| `percentage` | Integer | Tingkat asimetri (0-100%) |
| `severity` | String | Klasifikasi: "No Asymmetry", "Mild", "Moderate", "Severe" |
| `is_stroke_detected` | Boolean | True jika mendeteksi indikasi stroke |
| `mouth_diff` | Float | Perbedaan posisi mulut (pixel) |
| `eye_asymmetry` | Float | Tingkat asimetri mata (0-1) |
| `face_scale` | Float | Skala wajah dalam frame |
| `asymmetry_score` | Float | Score gabungan asimetri |
| `image_result` | String | Base64-encoded result image |

---

### 2️⃣ **POST** `/analyze/eye-symmetry`

Mendeteksi asimetri mata khusus (ptosis, eye droop).

#### Request

**Headers:**
```http
X-API-Key: your-api-key-from-.env
Content-Type: multipart/form-data
```

**Body:**
- `file` (File, required): Gambar wajah

#### Response Success (200 OK)

```json
{
  "status": "success",
  "analysis": {
    "left_eye_openness": 0.92,
    "right_eye_openness": 0.78,
    "eye_symmetry_score": 0.85,
    "severity": "Moderate Eye Asymmetry",
    "is_ptosis_detected": true,
    "eyelid_difference": 0.14,
    "asymmetry_percentage": 15
  },
  "image_result": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
}
```


## 7. Testing (Automated Tests)

### Menjalankan Tests

```bash
# Run all tests
pytest -v

# Run dengan coverage report
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_main.py -v
```

**Output yang diharapkan:**
```
tests/test_main.py::test_api_key_validation PASSED
tests/test_main.py::test_upload_valid_image PASSED
...
===================== 8 passed in 1.23s =====================
```

## 8. Deployment

### Quick Deployment (Development)

```bash
# Jangan gunakan ini untuk production!
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Deployment dengan Gunicorn

```bash
# Install Gunicorn
pip install gunicorn

# Run dengan multiple workers
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 9. Changelog

### v0.0.1 (2026-03-05)

- Initial release
- `/analyze/facial-palsy` endpoint
- `/analyze/eye-symmetry` endpoint
- API Key authentication
- Comprehensive documentation
- Unit tests

---

<div align="center">

**© 2026 Facial Palsy Detection API**

</div>
