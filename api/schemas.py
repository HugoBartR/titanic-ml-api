"""
Pydantic schemas for the Titanic Survival Prediction API.

This module defines the data models for API requests and responses,
ensuring proper validation and documentation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class Sex(str, Enum):
    """Passenger sex enumeration."""
    MALE = "male"
    FEMALE = "female"


class Embarked(str, Enum):
    """Port of embarkation enumeration."""
    S = "S"  # Southampton
    C = "C"  # Cherbourg
    Q = "Q"  # Queenstown


class Passenger(BaseModel):
    """Schema for a single passenger."""
    passenger_id: int = Field(..., description="Unique passenger identifier")
    pclass: int = Field(..., ge=1, le=3,
                        description="Passenger class (1=First, 2=Second, 3=Third)")
    sex: Sex = Field(..., description="Passenger sex")
    age: float = Field(..., ge=0, le=120, description="Passenger age")
    sibsp: int = Field(..., ge=0,
                       description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0,
                       description="Number of parents/children aboard")
    fare: float = Field(..., ge=0, description="Passenger fare")
    embarked: Embarked = Field(..., description="Port of embarkation")

    @validator('age')
    def validate_age(cls, v):
        """Validate age is reasonable."""
        if v < 0 or v > 120:
            raise ValueError('Age must be between 0 and 120')
        return v

    @validator('fare')
    def validate_fare(cls, v):
        """Validate fare is reasonable."""
        if v < 0:
            raise ValueError('Fare must be non-negative')
        return v

    class Config:
        schema_extra = {
            "example": {
                "passenger_id": 1,
                "pclass": 1,
                "sex": "female",
                "age": 29.0,
                "sibsp": 0,
                "parch": 0,
                "fare": 211.3375,
                "embarked": "S"
            }
        }


class Prediction(BaseModel):
    """Schema for a single prediction result."""
    passenger_id: int
    survived: bool
    survival_probability: float = Field(..., ge=0, le=1)

    class Config:
        schema_extra = {
            "example": {
                "passenger_id": 1,
                "survived": True,
                "survival_probability": 0.85
            }
        }


class PredictionRequest(BaseModel):
    """Schema for a single prediction request."""
    passenger: Passenger


class BatchPredictionRequest(BaseModel):
    """Schema for a batch prediction request."""
    passengers: List[Passenger] = Field(..., min_items=1, max_items=1000)


class PredictionResponse(BaseModel):
    """Schema for a single prediction response."""
    prediction: Prediction
    model_version: str
    prediction_time: float


class BatchPredictionResponse(BaseModel):
    """Schema for a batch prediction response."""
    predictions: List[Prediction]
    model_version: str
    prediction_time: float
    total_passengers: int


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    model_loaded: bool
    model_version: str
    uptime: float


class FeatureImportance(BaseModel):
    """Schema for feature importance response."""
    feature: str
    importance: float


class ModelInfoResponse(BaseModel):
    """Schema for model information response."""
    model_version: str
    feature_importance: List[FeatureImportance]
    model_metrics: Dict[str, float]
    training_date: str


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str
    detail: Optional[str] = None
    timestamp: str


class APIKeyHeader(BaseModel):
    """Schema for API key header."""
    x_api_key: Optional[str] = Field(None, alias="X-API-Key")


# Example data for testing
EXAMPLE_PASSENGER = {
    "passenger_id": 1,
    "pclass": 1,
    "sex": "female",
    "age": 29.0,
    "sibsp": 0,
    "parch": 0,
    "fare": 211.3375,
    "embarked": "S"
}

EXAMPLE_BATCH_REQUEST = {
    "passengers": [
        {
            "passenger_id": 1,
            "pclass": 1,
            "sex": "female",
            "age": 29.0,
            "sibsp": 0,
            "parch": 0,
            "fare": 211.3375,
            "embarked": "S"
        },
        {
            "passenger_id": 2,
            "pclass": 3,
            "sex": "male",
            "age": 23.0,
            "sibsp": 0,
            "parch": 0,
            "fare": 7.9250,
            "embarked": "S"
        }
    ]
}
