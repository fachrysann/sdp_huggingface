from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import dari module app
from app.config import ALLOWED_ORIGINS
from app.api.routes import router as api_router

app = FastAPI(title="Stroke Facial Analyzer API")

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
app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "Welcome to Stroke Assessment API. System is running!"}