import cv2
import numpy as np
import base64
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.schemas import StrokePredictorInput

from app.config import get_api_key, MAX_FILE_SIZE, ALLOWED_MIME_TYPES, ALLOWED_AUDIO_MIME_TYPES
from app.services.facial_service import FaceAnalyzerService
from app.services.riskometer_service import StrokePredictorService
from app.services.speech_service import AudioAnalyzerService

router = APIRouter()
analyzer_instance = FaceAnalyzerService()
predictor_instance = StrokePredictorService()
audio_instance = AudioAnalyzerService()

def get_analyzer():
    return analyzer_instance

def get_predictor(): 
    return predictor_instance

def get_audio_analyzer():
    return audio_instance

# --- HELPER FUNCTION: Untuk mengolah gambar yang diupload ---
async def process_uploaded_image(file: UploadFile):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported media type. Only JPEG/PNG are allowed.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Payload too large. File exceeds maximum allowed size.")

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
        raise HTTPException(status_code=413, detail="Payload too large. File exceeds maximum allowed size.")
    return contents

# --- HELPER FUNCTION: Kompresi Base64 Khusus Mobile ---
def encode_image_for_mobile(img_array, max_width=720, jpeg_quality=75):
    """
    Me-resize gambar jika terlalu besar dan mengompresnya ke JPEG
    agar teks Base64 tidak membebani aplikasi mobile.
    """
    h, w = img_array.shape[:2]
    
    # 1. Resize jika lebar gambar melebihi batas maksimal (misal 720px)
    if w > max_width:
        ratio = max_width / float(w)
        new_h = int(h * ratio)
        img_array = cv2.resize(img_array, (max_width, new_h), interpolation=cv2.INTER_AREA)
    
    # 2. Kompres ke JPEG (mengurangi ukuran file tanpa merusak visualisasi garis/titik)
    encode_param =[int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, buffer = cv2.imencode('.jpg', img_array, encode_param)
    
    # 3. Ubah ke teks Base64
    return base64.b64encode(buffer).decode('utf-8')

# ==================================================
# ENDPOINT 1: FACIAL PALSY (Deteksi Senyum/Mulut)
# ==================================================
@router.post(
    "/analyze/facial-palsy",
    tags=["Facial Symmetry Analysis"],
    summary="Detect Facial Palsy (Face Symmetry)",
    description="Uploads a facial image and analyzes it for asymmetric features related to facial palsy, particularly around the mouth and jawline. Returns the severity score and a base64 encoded visualization.",
    responses={
        200: {"description": "Successfully analyzed the face."},
        400: {"description": "Invalid or corrupted image data."},
        403: {"description": "Forbidden: Invalid or missing API Key."},
        413: {"description": "Payload too large (File exceeds 5MB)."},
        415: {"description": "Unsupported media type (Only JPEG/PNG allowed)."},
        500: {"description": "Internal server error during analysis."}
    }
)
async def analyze_facial_palsy(
    file: UploadFile = File(..., description="Image file (JPEG, PNG) containing a clear view of the patient's face"), 
    api_key: str = Depends(get_api_key),
    analyzer: FaceAnalyzerService = Depends(get_analyzer)
):
    img = await process_uploaded_image(file)

    try:
        results, processed_img = analyzer.analyze_facial_palsy(img)
        if results is None:
            return {"status": "error", "message": "No face detected"}

        # _, buffer = cv2.imencode('.jpg', processed_img)
        # img_base64 = base64.b64encode(buffer).decode('utf-8')

        img_base64 = encode_image_for_mobile(processed_img)

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
@router.post(
    "/analyze/eye-symmetry",
    tags=["Facial Symmetry Analysis"],
    summary="Analyze Eye Symmetry (Gaze Detection)",
    description="Uploads an image to evaluate the symmetry of eye gaze and pupil coordination. Helps in detecting asymmetrical eye movements associated with strokes.",
    responses={
        200: {"description": "Successfully analyzed eye symmetry."},
        400: {"description": "Invalid or corrupted image data."},
        403: {"description": "Forbidden: Invalid or missing API Key."},
        413: {"description": "Payload too large (File exceeds 5MB)."},
        415: {"description": "Unsupported media type (Only JPEG/PNG allowed)."},
        500: {"description": "Internal server error during analysis."}
    }
)
async def analyze_eye_symmetry(
    file: UploadFile = File(..., description="Image file (JPEG, PNG) with clear visibility of both eyes"), 
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
        # _, buffer = cv2.imencode('.jpg', processed_img)
        # img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_base64 = encode_image_for_mobile(processed_img)

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
@router.post(
    "/predict/riskometer",
    tags=["Riskometer Prediction"],
    summary="Predict Stroke Risk from Patient Data",
    description="Receives tabular patient metadata (Age, BMI, Glucose Levels, etc.) and runs it through a Random Forest Machine Learning model to calculate the probability of a stroke.",
    responses={
        200: {"description": "Successfully predicted stroke risk."},
        400: {"description": "Bad Request: Missing or invalid required features (e.g., invalid gender value)."},
        403: {"description": "Forbidden: Invalid or missing API Key."},
        422: {"description": "Validation Error: Incorrect data types provided in the JSON body."},
        500: {"description": "Internal server error during ML model prediction."}
    }
)
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
        raise HTTPException(status_code=500, detail=f"Error ML: {str(e)}")

# ==================================================
# ENDPOINT 4: AUDIO CLASSIFICATION (DYSARTHRIA)
# ==================================================
@router.post(
    "/analyze/speech-dysarthria",
    tags=["Speech Dysarthria Analysis"],
    summary="Detect Dysarthria from Speech",
    description="Uploads a voice recording (.wav) to detect signs of Dysarthria (slurred speech) often associated with strokes. Converts audio to Mel-Spectrogram and runs it through a ResNet-18 model.",
    responses={
        200: {"description": "Successfully analyzed speech recording."},
        403: {"description": "Forbidden: Invalid or missing API Key."},
        413: {"description": "Payload too large (File exceeds 5MB)."},
        415: {"description": "Unsupported media type (Only .wav allowed)."},
        500: {"description": "Internal server error processing the audio."}
    }
)
async def analyze_speech(
    file: UploadFile = File(..., description="Audio file in .wav format containing the patient's speech"),
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