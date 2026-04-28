import cv2
import numpy as np
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.schemas import StrokePredictorInput
from datetime import datetime
import tempfile # Tambahkan di atas
import os

from app.config import (
    get_api_key, MAX_FILE_SIZE, ALLOWED_MIME_TYPES, ALLOWED_AUDIO_MIME_TYPES, ALLOWED_VIDEO_MIME_TYPES,
    supabase, SUPABASE_BUCKET_NAME
)

from app.services.facial_service import FaceAnalyzerService
from app.services.riskometer_service import StrokePredictorService
from app.services.speech_service import AudioAnalyzerService
from app.services.arm_service import ArmAnalyzerService 

router = APIRouter()
analyzer_instance = FaceAnalyzerService()
predictor_instance = StrokePredictorService()
audio_instance = AudioAnalyzerService()
arm_instance = ArmAnalyzerService() 

def get_analyzer():
    return analyzer_instance

def get_predictor(): 
    return predictor_instance

def get_audio_analyzer():
    return audio_instance

def get_arm_analyzer():
    return arm_instance

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

# ==================================================
# UPLOAD TO SUPABASE
# ==================================================
def upload_image_to_supabase(img_array, folder_name: str, max_width=1080, jpeg_quality=95):
    """
    Me-resize gambar, kompres ke JPEG, lalu upload ke Supabase Storage.
    Mengembalikan Public URL.
    """
    h, w = img_array.shape[:2]
    
    # 1. Resize
    if w > max_width:
        ratio = max_width / float(w)
        new_h = int(h * ratio)
        img_array = cv2.resize(img_array, (max_width, new_h), interpolation=cv2.INTER_AREA)
    
    # 2. Encode to JPEG bytes
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, buffer = cv2.imencode('.jpg', img_array, encode_param)
    image_bytes = buffer.tobytes()
    
    # 3. Generate Unique Filename
    filename = f"{folder_name}/{uuid.uuid4().hex}.jpg"
    
    try:
        # 4. Upload to Supabase Bucket
        supabase.storage.from_(SUPABASE_BUCKET_NAME).upload(
            path=filename,
            file=image_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        
        # 5. Get Public URL
        public_url = supabase.storage.from_(SUPABASE_BUCKET_NAME).get_public_url(filename)
        return public_url
    except Exception as e:
        print(f"Supabase Upload Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Gagal mengunggah gambar ke Storage.")
    
def upload_video_to_supabase(video_path: str, folder_name: str):
    """
    Mengupload video (mp4) ke Supabase Storage dan mengembalikan Public URL.
    """
    filename = f"{folder_name}/{uuid.uuid4().hex}.mp4"
    try:
        with open(video_path, "rb") as f:
            supabase.storage.from_(SUPABASE_BUCKET_NAME).upload(
                path=filename,
                file=f,
                file_options={"content-type": "video/mp4"}
            )
        public_url = supabase.storage.from_(SUPABASE_BUCKET_NAME).get_public_url(filename)
        return public_url
    except Exception as e:
        print(f"Supabase Video Upload Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Gagal mengunggah video ke Storage.")

# ==================================================
# LOGGING MLOPS TO SUPABASE
# ==================================================
def log_prediction_to_supabase(endpoint_name: str, input_data: dict, prediction_result: dict, media_url: str = None):
    """
    Menyimpan log prediksi ke Supabase untuk keperluan MLOps (Data Tracking)
    """
    try:
        log_data = {
            "endpoint": endpoint_name,
            "input_data": input_data,
            "media_url": media_url,
            "severity_score": prediction_result.get("severity_score"),
            "status_label": prediction_result.get("status_label"),
            "metrics": prediction_result.get("metrics"),
            "human_feedback": None # Disiapkan untuk Tahap Feedback Loop
        }
        
        # Insert ke tabel yang baru kita buat
        supabase.table("prediction_logs").insert(log_data).execute()
        print(f"MLOps Log saved for {endpoint_name}")
        
    except Exception as e:
        # Kita hanya print errornya, jangan di-raise HTTPException
        # agar user tetap mendapatkan hasil prediksi walaupun sistem log gagal
        print(f"MLOps Logging Error: {str(e)}")

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

        img_url = upload_image_to_supabase(processed_img, folder_name="facial_palsy")

        log_prediction_to_supabase(
            endpoint_name="facial_palsy",
            input_data={"filename": file.filename},
            prediction_result=results,
            media_url=img_url
        )

        return {
            "status": "success",
            "analysis": results,
            "image_url": img_url 
        }
    except HTTPException:
        raise    
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
        img_url = upload_image_to_supabase(processed_img, folder_name="eye_symmetry")

        log_prediction_to_supabase(
            endpoint_name="eye_symmetry",
            input_data={"filename": file.filename},
            prediction_result=results,
            media_url=img_url
        )

        return {
            "status": "success",
            "analysis": results,
            "image_url": img_url  # Changed from image_result to image_url
        }
    except HTTPException:
        raise    
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

        log_prediction_to_supabase(
            endpoint_name="riskometer",
            input_data=data.model_dump(), # Simpan data gender, bmi, glukosa, dll
            prediction_result=result,
            media_url=None # Tabular tidak ada gambar
        )
        
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

        log_prediction_to_supabase(
            endpoint_name="speech_dysarthria",
            input_data={"filename": file.filename},
            prediction_result=result,
            media_url=None # Nanti bisa diupdate jika audio di-upload ke Supabase bucket
        )
        
        return {
            "status": "success",
            "filename": file.filename,
            "data": result
        }
    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses audio: {str(e)}")
    
# ==================================================
# ENDPOINT 5: ARM WEAKNESS (Deteksi Kelemahan Lengan)
# ==================================================
@router.post(
    "/analyze/arm-weakness",
    tags=["Arm Weakness Analysis"],
    summary="Analyze Arm Weakness from Video",
    description="Uploads a video evaluating the patient's arm strength by holding both arms raised. Returns severity score and a processed mp4 video URL.",
)
async def analyze_arm_weakness(
    file: UploadFile = File(..., description="Video file (.mp4) of the patient holding both arms up."),
    api_key: str = Depends(get_api_key),
    analyzer: ArmAnalyzerService = Depends(get_arm_analyzer)
):
    # 1. Validasi File
    if file.content_type not in ALLOWED_VIDEO_MIME_TYPES and not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=415, detail="Unsupported media type. Only MP4/AVI/MOV are allowed.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Payload too large. File exceeds maximum allowed size.")

    # 2. Buat Temporary File untuk Input dan Output (cv2 membutuhkan path file asli)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_in:
            temp_in.write(contents)
            input_path = temp_in.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_out:
            output_path = temp_out.name

        # 3. Proses Analisis Video
        results = analyzer.analyze_arm_weakness(input_path, output_path)

        # 4. Upload Hasil Video ke Supabase
        video_url = upload_video_to_supabase(output_path, folder_name="arm_weakness")

        # 5. Bersihkan Temp File
        os.remove(input_path)
        os.remove(output_path)

        # 6. Logging ke MLOps
        log_prediction_to_supabase(
            endpoint_name="arm_weakness",
            input_data={"filename": file.filename},
            prediction_result=results,
            media_url=video_url
        )

        return {
            "status": "success",
            "analysis": results,
            "video_url": video_url 
        }

    except Exception as e:
        print(f"Server Error (Arm Weakness): {str(e)}")
        # Pastikan file temporary dihapus walaupun terjadi error
        if 'input_path' in locals() and os.path.exists(input_path): os.remove(input_path)
        if 'output_path' in locals() and os.path.exists(output_path): os.remove(output_path)
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")