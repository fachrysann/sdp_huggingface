import io
from fastapi.testclient import TestClient
from app.main import app
from app.config import API_KEY
import cv2
import numpy as np

client = TestClient(app)

# --- SKENARIO 1: Keamanan API Key ---
def test_analyze_without_api_key():
    """Test: API harus menolak request jika tidak ada API Key"""
    # Bikin dummy file gambar
    dummy_file = {"file": ("test.jpg", b"dummy_image_data", "image/jpeg")}
    
    # Hit API tanpa header X-API-Key
    response = client.post("/analyze", files=dummy_file)
    
    assert response.status_code == 403
    assert response.json() == {"detail": "Forbidden: Invalid or missing API Key"}

# --- SKENARIO 2: Validasi Format File ---
def test_analyze_with_invalid_file_type():
    """Test: API harus menolak jika yang diupload bukan gambar"""
    # Bikin dummy file text (bukan gambar)
    dummy_file = {"file": ("test.txt", b"ini isi text", "text/plain")}
    headers = {"X-API-Key": API_KEY}
    
    response = client.post("/analyze", files=dummy_file, headers=headers)
    
    assert response.status_code == 415
    assert "Unsupported media type" in response.json()["detail"]

# --- SKENARIO 3: Logic MediaPipe (Gambar valid, tapi tidak ada wajah) ---
def test_analyze_no_face_detected():
    """Test: API harus mengembalikan status error/no face jika gambar kosong/hitam"""
    
    # Bikin gambar hitam kosong (100x100 pixel) menggunakan numpy & OpenCV
    blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', blank_image)
    image_bytes = buffer.tobytes()

    # Siapkan payload
    dummy_file = {"file": ("blank.jpg", image_bytes, "image/jpeg")}
    headers = {"X-API-Key": API_KEY}

    # Hit API
    response = client.post("/analyze", files=dummy_file, headers=headers)
    
    # Karena API sukses memproses (hanya saja tidak ada wajah), statusnya harus 200 OK
    assert response.status_code == 200
    
    # Pastikan respon dari ml_service.py sesuai
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "No face detected"