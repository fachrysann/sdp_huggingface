import json
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

class StrokePredictorInput(BaseModel):
    gender: int = Field(..., description="0 for Female, 1 for Male")
    age: int
    hypertension: int = Field(..., description="0 for No, 1 for Yes")
    heart_disease: int = Field(..., description="0 for No, 1 for Yes")
    ever_married: int = Field(..., description="0 for No, 1 for Yes")
    avg_glucose_level: float = Field(..., description="Average glucose level in blood")
    bmi: float = Field(..., description="Body Mass Index")
    smoking_status: int = Field(..., description="0 for No, 1 for Yes")
    
    model_config = {
        "json_schema_extra": {
            "examples":[
                {
                    "gender": 1,
                    "age": 65,
                    "hypertension": 1,
                    "heart_disease": 0,
                    "ever_married": 1,
                    "avg_glucose_level": 228.69,
                    "bmi": 36.6,
                    "smoking_status": 1
                }
            ]
        }
    }

def configure_openapi_schemas(app: FastAPI):
    
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=app.openapi_tags,
        )

        schema_str = json.dumps(openapi_schema)

        # Kamus Find & Replace
        replacements = {
            "Body_analyze_facial_palsy_api_v1_analyze_facial_palsy_post": "FacialPalsyUpload",
            "Body_analyze_eye_symmetry_api_v1_analyze_eye_symmetry_post": "EyeSymmetryUpload",
            "Body_analyze_speech_api_v1_analyze_speech_dysarthria_post": "SpeechAudioUpload",
        }

        for old_name, new_name in replacements.items():
            schema_str = schema_str.replace(old_name, new_name)

        app.openapi_schema = json.loads(schema_str)
        return app.openapi_schema

    app.openapi = custom_openapi