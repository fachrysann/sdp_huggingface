import os
from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from supabase import create_client, Client

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("FATAL ERROR: API_KEY tidak ditemukan di file .env!")

# ==========================================
# SETUP SUPABASE
# ==========================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "stroke_images")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("FATAL ERROR: SUPABASE_URL atau SUPABASE_KEY tidak ditemukan di .env!")

# Inisialisasi Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# ==========================================

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", 25)) * 1024 * 1024
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/jpg"]
ALLOWED_AUDIO_MIME_TYPES =["audio/wav", "audio/x-wav", "audio/wave"]
ALLOWED_VIDEO_MIME_TYPES =["video/mp4", "video/x-msvideo", "video/quicktime"]

api_key_header = APIKeyHeader(
    name="X-API-Key", 
    auto_error=False,
    description="API Key required for accessing the endpoints. Place it in the 'X-API-Key' header."
)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing API Key")