import os
from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("FATAL ERROR: API_KEY tidak ditemukan di file .env!")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", 5)) * 1024 * 1024
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/jpg"]
ALLOWED_AUDIO_MIME_TYPES =["audio/wav", "audio/x-wav", "audio/wave"]

api_key_header = APIKeyHeader(
    name="X-API-Key", 
    auto_error=False,
    description="API Key required for accessing the endpoints. Place it in the 'X-API-Key' header."
)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing API Key")