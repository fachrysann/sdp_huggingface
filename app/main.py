from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import dari module app
from app.config import ALLOWED_ORIGINS
from app.api.routes import router as api_router

tags_metadata =[
    {"name": "Facial Symmetry Analysis", "description": "Endpoints for facial palsy and eye symmetry detection."},
    {"name": "Riskometer Prediction", "description": "Endpoints for stroke risk prediction using patient data."},
    {"name": "Speech Dysarthria Analysis", "description": "Endpoints for detecting Dysarthria from voice recordings."}
]

app = FastAPI(
    title="Stroke Detect Pro API",
    description="An AI-powered API to assist in assessing Stroke symptoms including Facial Palsy, Eye Symmetry, Speech Dysarthria, and Tabular Risk prediction.",
    version="1.0.0",
    openapi_tags=tags_metadata,
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["X-API-Key", "Content-Type"],
)

# Sambungkan rute dari routes.py
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Welcome to Stroke Assessment API. System is running!",
        "version": "v1.0.0",
        "docs_url": "/docs" # Memberi tahu front-end di mana letak dokumentasinya
    }