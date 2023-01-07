from typing import Optional

from pydantic import BaseModel, Field


class CustomerRequest(BaseModel):
    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = Field(default=0, ge=0, le=1)
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    tenure: Optional[float] = None
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: Optional[str] = None
    PaperlessBilling: Optional[str] = None
    PaymentMethod: Optional[str] = None
    MonthlyCharges: Optional[float] = None
    TotalCharges: Optional[float] = None


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    threshold_used: float
