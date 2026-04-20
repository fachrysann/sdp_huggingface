import cv2
import numpy as np
import base64
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.schemas import StrokePredictorInput

from app.config import get_api_key, MAX_FILE_SIZE, ALLOWED_MIME_TYPES, ALLOWED_AUDIO_MIME_TYPES
from app.services.vision_service import FaceAnalyzerService
from app.services.tabular_service import StrokePredictorService
from app.services.audio_service import AudioAnalyzerService

router = APIRouter()
analyzer_instance = FaceAnalyzerService()
predictor_instance = StrokePredictorService()
audio_instance = AudioAnalyzerService()

def get_analyzer():
    return analyzer_instance

def get_predictor(): # <--- Tambahkan ini
    return predictor_instance

def get_audio_analyzer():
    return audio_instance

# --- HELPER FUNCTION: Untuk mengolah gambar yang diupload ---
async def process_uploaded_image(file: UploadFile):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported media type.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Payload too large.")

    try:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: raise ValueError
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image data.")
    
# --- HELPER FUNCTION: Untuk mengolah audio yang diupload ---
async def process_uploaded_audio(file: UploadFile):
    # Validasi Tipe File Ekstensi & MIME
    if not file.filename.lower().endswith('.wav') or file.content_type not in ALLOWED_AUDIO_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported media type. Hanya mendukung file .wav")

    # Validasi Ukuran & Baca File
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Payload too large.")
    return contents


# ==================================================
# ENDPOINT 1: FACIAL PALSY (Deteksi Senyum/Mulut)
# ==================================================
@router.post("/analyze/facial-palsy")
async def analyze_facial_palsy(
    file: UploadFile = File(...), 
    api_key: str = Depends(get_api_key),
    analyzer: FaceAnalyzerService = Depends(get_analyzer)
):
    img = await process_uploaded_image(file)

    try:
        results, processed_img = analyzer.analyze_facial_palsy(img)
        if results is None:
            return {"status": "error", "message": "No face detected"}

        _, buffer = cv2.imencode('.jpg', processed_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "status": "success",
            "analysis": results,
            "image_result": f"data:image/jpeg;base64,{img_base64}"
        }
    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during processing.")


# ==================================================
# ENDPOINT 2: EYE SYMMETRY (Deteksi Lirikan Mata)
# ==================================================
@router.post("/analyze/eye-symmetry")
async def analyze_eye_symmetry(
    file: UploadFile = File(...), 
    api_key: str = Depends(get_api_key),
    analyzer: FaceAnalyzerService = Depends(get_analyzer)
):
    # 1. Validasi & Baca Gambar
    img = await process_uploaded_image(file)

    try:
        # 2. Proses menggunakan Method baru di Class
        results, processed_img = analyzer.analyze_eye_symmetry(img)

        if results is None:
            return {"status": "error", "message": "No face detected"}

        # 3. Encode hasil ke Base64
        _, buffer = cv2.imencode('.jpg', processed_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 4. Kembalikan Response
        return {
            "status": "success",
            "analysis": results,
            "image_result": f"data:image/jpeg;base64,{img_base64}"
        }
    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during processing.")
    
# ==================================================
# ENDPOINT 3: TABULAR STROKE PREDICTION
# ==================================================
@router.post("/predict/tabular-data")
async def predict_stroke(
    data: StrokePredictorInput, 
    api_key: str = Depends(get_api_key),
    predictor: StrokePredictorService = Depends(get_predictor)
):
    try:
        # Panggil service prediksi
        result = predictor.predict_stroke(data.model_dump())
        
        return {
            "status": "success",
            "data": result
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Server Error: {str(e)}")
        # UBAH BARIS DI BAWAH INI SEMENTARA UNTUK MELIHAT ERROR
        raise HTTPException(status_code=500, detail=f"Error ML: {str(e)}")
    
# ==================================================
# ENDPOINT 4: AUDIO CLASSIFICATION (DYSARTHRIA)
# ==================================================
@router.post("/analyze/speech")
async def analyze_speech(
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key),
    audio_analyzer: AudioAnalyzerService = Depends(get_audio_analyzer)
):
    # Validasi Ukuran & Baca File
    contents = await process_uploaded_audio(file)

    try:
        # Jalankan prediksi
        result = audio_analyzer.predict_audio(contents)
        
        return {
            "status": "success",
            "filename": file.filename,
            "data": result
        }
    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses audio: {str(e)}")