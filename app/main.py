from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import dari module app
from app.config import ALLOWED_ORIGINS
from app.api.routes import router as api_router
from app.schemas import configure_openapi_schemas 

tags_metadata =[
    {"name": "Facial Symmetry Analysis", "description": "Endpoints for facial palsy and eye symmetry detection."},
    {"name": "Arm Weakness Analysis", "description": "Endpoints for detecting motor weakness from videos."},
    {"name": "Riskometer Prediction", "description": "Endpoints for stroke risk prediction using patient data."},
    {"name": "Speech Dysarthria Analysis", "description": "Endpoints for detecting Dysarthria from voice recordings."}
]

app = FastAPI(
    title="Stroke Detect Pro API",
    description="An AI-powered API to assist in assessing Stroke symptoms for Stroke Detect Pro app. Provides endpoints for facial palsy analysis, eye symmetry detection, stroke risk prediction from patient data, and Dysarthria detection from speech.",
    version="1.0.0",
    openapi_tags=tags_metadata,
)

# Terapkan perapian Schemas
configure_openapi_schemas(app)

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
        "message": "Welcome to Stroke Detect Pro API. System is running!",
        "version": "v1.0.0",
        "docs_url": "/docs" 
    }