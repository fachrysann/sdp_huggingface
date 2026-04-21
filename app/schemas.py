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