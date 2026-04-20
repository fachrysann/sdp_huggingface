from pydantic import BaseModel, Field

class StrokePredictorInput(BaseModel):
    gender: int = Field(..., description="0 for Female, 1 for Male")
    age: int
    hypertension: int = Field(..., description="0 for No, 1 for Yes")
    heart_disease: int = Field(..., description="0 for No, 1 for Yes")
    ever_married: int = Field(..., description="0 for No, 1 for Yes")
    avg_glucose_level: float
    bmi: float
    smoking_status: int = Field(..., description="0 for No, 1 for Yes")