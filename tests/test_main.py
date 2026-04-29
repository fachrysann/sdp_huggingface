import io
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.config import API_KEY
from app.config import MAX_FILE_SIZE

client = TestClient(app)

# ==========================================
# SKENARIO 0: ROOT ENDPOINT
# ==========================================
def test_read_root():
    """Test: API Root harus merespons dengan pesan selamat datang"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Stroke Detect Pro API" in data["message"]


# ==========================================
# SKENARIO 1: FACIAL PALSY ENDPOINT
# ==========================================
def test_facial_palsy_without_api_key():
    """Test: API harus menolak request jika tidak ada API Key"""
    dummy_file = {"file": ("test.jpg", b"dummy_image_data", "image/jpeg")}
    response = client.post("/api/v1/analyze/facial-palsy", files=dummy_file)
    
    assert response.status_code == 403
    assert response.json() == {"detail": "Forbidden: Invalid or missing API Key"}

def test_facial_palsy_invalid_file_type():
    """Test: API harus menolak jika yang diupload bukan gambar"""
    dummy_file = {"file": ("test.txt", b"ini isi text", "text/plain")}
    headers = {"X-API-Key": API_KEY}
    
    response = client.post("/api/v1/analyze/facial-palsy", files=dummy_file, headers=headers)
    
    assert response.status_code == 415
    assert "Unsupported media type" in response.json()["detail"]

def test_facial_palsy_no_face_detected():
    """Test: API harus mengembalikan status error jika gambar kosong/hitam (tidak ada wajah)"""
    # Bikin gambar hitam kosong (100x100 pixel)
    blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', blank_image)
    image_bytes = buffer.tobytes()

    dummy_file = {"file": ("blank.jpg", image_bytes, "image/jpeg")}
    headers = {"X-API-Key": API_KEY}

    response = client.post("/api/v1/analyze/facial-palsy", files=dummy_file, headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "No face detected"


# ==========================================
# SKENARIO 2: EYE SYMMETRY ENDPOINT
# ==========================================
def test_eye_symmetry_no_face_detected():
    """Test: Eye Symmetry harus menghandle gambar yang tidak memuat wajah dengan aman"""
    blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', blank_image)
    
    dummy_file = {"file": ("blank.jpg", buffer.tobytes(), "image/jpeg")}
    headers = {"X-API-Key": API_KEY}

    response = client.post("/api/v1/analyze/eye-symmetry", files=dummy_file, headers=headers)
    
    assert response.status_code == 200
    assert response.json()["message"] == "No face detected"


# ==========================================
# SKENARIO 3: RISKOMETER ENDPOINT (TABULAR)
# ==========================================
def test_riskometer_missing_fields():
    """Test: API Riskometer menolak jika payload JSON tidak lengkap berdasarkan Pydantic Schema"""
    headers = {"X-API-Key": API_KEY}
    # Hanya mengirim gender dan age (kurang field wajib lainnya)
    incomplete_data = {"gender": 1, "age": 65}
    
    response = client.post("/api/v1/predict/riskometer", json=incomplete_data, headers=headers)
    
    assert response.status_code == 422 # 422 Unprocessable Entity dari Pydantic

def test_riskometer_invalid_gender_value():
    """Test: Validasi manual di service akan menolak jika nilai gender diluar 0 atau 1"""
    headers = {"X-API-Key": API_KEY}
    invalid_data = {
        "gender": 3, # Invalid (hanya menerima 0 atau 1)
        "age": 65,
        "hypertension": 1,
        "heart_disease": 0,
        "ever_married": 1,
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": 1
    }
    
    response = client.post("/api/v1/predict/riskometer", json=invalid_data, headers=headers)
    
    assert response.status_code == 400
    assert "Invalid gender" in response.json()["detail"]


# ==========================================
# SKENARIO 4: SPEECH DYSARTHRIA ENDPOINT
# ==========================================
def test_speech_invalid_file_type():
    """Test: API Speech harus menolak jika file bukan audio (.wav / .m4a)"""
    dummy_file = {"file": ("video.mp4", b"dummy video bytes", "video/mp4")}
    headers = {"X-API-Key": API_KEY}

    response = client.post("/api/v1/analyze/speech-dysarthria", files=dummy_file, headers=headers)
    
    assert response.status_code == 415
    assert "Unsupported media type" in response.json()["detail"]

def test_speech_invalid_extension_valid_mime():
    """Test: API menolak jika mime audio benar tapi ekstensinya tidak didukung (.mp3)"""
    dummy_file = {"file": ("audio.mp3", b"dummy audio", "audio/mp3")}
    headers = {"X-API-Key": API_KEY}

    response = client.post("/api/v1/analyze/speech-dysarthria", files=dummy_file, headers=headers)
    
    assert response.status_code == 415
    assert "Hanya mendukung file .wav dan .m4a" in response.json()["detail"]


# ==========================================
# SKENARIO 5: ARM WEAKNESS ENDPOINT
# ==========================================
def test_arm_weakness_invalid_file_type():
    """Test: API Arm Weakness harus menolak jika dikirim gambar/dokumen (bukan video)"""
    dummy_file = {"file": ("image.jpg", b"dummy image", "image/jpeg")}
    headers = {"X-API-Key": API_KEY}

    response = client.post("/api/v1/analyze/arm-weakness", files=dummy_file, headers=headers)
    
    assert response.status_code == 415
    assert "Unsupported media type" in response.json()["detail"]

def test_arm_weakness_without_api_key():
    """Test: Akses video endpoint dibatasi tanpa header Auth"""
    dummy_file = {"file": ("test.mp4", b"dummy video", "video/mp4")}
    
    response = client.post("/api/v1/analyze/arm-weakness", files=dummy_file)
    
    assert response.status_code == 403

# ==========================================
# SKENARIO 6: KEAMANAN & MIDDLEWARE (CORS & AUTH)
# ==========================================

def test_invalid_api_key():
    """Test: API harus menolak dengan 403 jika API Key yang diberikan SALAH"""
    dummy_file = {"file": ("test.jpg", b"dummy", "image/jpeg")}
    headers = {"X-API-Key": "TENTU_SAJA_INI_KEY_YANG_SALAH_123"}
    
    response = client.post("/api/v1/analyze/facial-palsy", files=dummy_file, headers=headers)
    
    assert response.status_code == 403
    assert response.json()["detail"] == "Forbidden: Invalid or missing API Key"

def test_cors_preflight():
    """Test: Memastikan CORS Middleware mengizinkan preflight request (OPTIONS) dari browser frontend"""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "X-API-Key"
    }
    response = client.options("/api/v1/predict/riskometer", headers=headers)
    
    # Preflight request selalu mengembalikan 200 OK jika diizinkan
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "*" # Berdasarkan setting allow_origins=["*"]


# ==========================================
# SKENARIO 7: FILE SIZE LIMIT (PAYLOAD TERLALU BESAR)
# ==========================================

def test_image_payload_too_large():
    """Test: API menolak jika ukuran file melebihi MAX_FILE_SIZE"""
    headers = {"X-API-Key": API_KEY}
    
    # Membuat file dummy lebih besar 1 byte dari MAX_FILE_SIZE (default 25MB)
    oversized_data = b"0" * (MAX_FILE_SIZE + 1)
    dummy_file = {"file": ("big_image.jpg", oversized_data, "image/jpeg")}
    
    response = client.post("/api/v1/analyze/facial-palsy", files=dummy_file, headers=headers)
    
    assert response.status_code == 413
    assert "Payload too large" in response.json()["detail"]


# ==========================================
# SKENARIO 8: CORRUPTED DATA / FILE RUSAK
# ==========================================

def test_corrupted_image_data():
    """Test: File ekstensinya .jpg tapi isinya adalah teks acak (Corrupted Image)"""
    headers = {"X-API-Key": API_KEY}
    corrupted_data = b"Ini adalah teks biasa, bukan byte gambar valid"
    
    dummy_file = {"file": ("corrupted.jpg", corrupted_data, "image/jpeg")}
    response = client.post("/api/v1/analyze/eye-symmetry", files=dummy_file, headers=headers)
    
    # Exception dari OpenCV (cv2.imdecode gagal) akan ditangkap helper menjadi 400 Bad Request
    assert response.status_code == 400
    assert "Invalid or corrupted image data" in response.json()["detail"]

def test_corrupted_audio_data():
    """Test: File ekstensinya .wav tapi byte-nya rusak. Pydub/Soundfile harus gagal dengan aman."""
    headers = {"X-API-Key": API_KEY}
    corrupted_audio = b"Bukan RIFF file format yang valid"
    
    dummy_file = {"file": ("fake_audio.wav", corrupted_audio, "audio/wav")}
    response = client.post("/api/v1/analyze/speech-dysarthria", files=dummy_file, headers=headers)
    
    # Error pemrosesan ML (soundfile fail to read) mengembalikan 500 Internal Server Error
    assert response.status_code == 500
    assert "Terjadi kesalahan saat memproses audio" in response.json()["detail"]


# ==========================================
# SKENARIO 9: HAPPY PATH - RISKOMETER
# ==========================================

def test_riskometer_happy_path():
    """
    Test: Mengecek input valid pada ML Tabular Riskometer.
    Karena tidak butuh proses upload ke Supabase Bucket (Tabular only),
    tes ini bisa di run dan harus merespon success 200 beserta struktur output prediksi yang valid.
    """
    headers = {"X-API-Key": API_KEY}
    
    # Data pasien yang valid sesuai skema Pydantic
    valid_patient_data = {
        "gender": 1,
        "age": 65,
        "hypertension": 1,
        "heart_disease": 0,
        "ever_married": 1,
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": 1
    }
    
    response = client.post("/api/v1/predict/riskometer", json=valid_patient_data, headers=headers)
    
    # Cek HTTP Status
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    
    # Cek apakah response structure dari model ML valid
    prediction = data["data"]
    assert "severity_score" in prediction
    assert "status_label" in prediction
    assert "metrics" in prediction
    assert "risk_percentage" in prediction["metrics"]
    assert type(prediction["severity_score"]) == int