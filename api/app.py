"""
Titanic Survival Prediction API

A FastAPI-based REST API for predicting Titanic passenger survival.
Includes comprehensive error handling, logging, monitoring, and security.
"""

import time
import os
from typing import List, Optional
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

import structlog

from .schemas import (
    Passenger, Prediction, PredictionRequest, BatchPredictionRequest,
    PredictionResponse, BatchPredictionResponse, HealthResponse,
    ModelInfoResponse, ErrorResponse, APIKeyHeader
)
from .utils import (
    get_model_manager, get_security_manager, get_metrics_collector,
    validate_passenger_data, format_error_response
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Titanic Survival Prediction API")

    # Load model
    model_manager = get_model_manager()
    if not model_manager.load_model():
        logger.error("Failed to load model during startup")
        raise RuntimeError("Model loading failed")

    logger.info("API startup completed")
    yield

    # Shutdown
    logger.info("Shutting down Titanic Survival Prediction API")


# Create FastAPI app
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="A machine learning API for predicting Titanic passenger survival",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency for API key validation
async def validate_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Validate API key if required."""
    security_manager = get_security_manager()
    if not security_manager.validate_api_key(x_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return x_api_key


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.warning("Validation error", errors=exc.errors())
    return JSONResponse(
        status_code=422,
        content=format_error_response(
            "Validation error",
            str(exc.errors())
        )
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    logger.warning("HTTP exception",
                   status_code=exc.status_code, detail=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=format_error_response(exc.detail)
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content=format_error_response("Internal server error")
    )


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses."""
    start_time = time.time()

    # Log request
    logger.info("Request started",
                method=request.method,
                url=str(request.url),
                client_ip=request.client.host if request.client else None)

    # Process request
    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info("Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=process_time)

    return response


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    model_manager = get_model_manager()
    metrics_collector = get_metrics_collector()

    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.is_loaded(),
        model_version=model_manager.model_version,
        uptime=metrics_collector.get_metrics()["uptime"]
    )


# Root endpoint
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Titanic Survival Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Single prediction endpoint
@app.post("/predict",
          response_model=PredictionResponse,
          dependencies=[Depends(validate_api_key)],
          tags=["Predictions"])
async def predict_survival(request: PredictionRequest):
    """Predict survival for a single passenger."""
    start_time = time.time()

    try:
        model_manager = get_model_manager()
        metrics_collector = get_metrics_collector()

        if not model_manager.is_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert passenger to dict for processing
        passenger_data = request.passenger.dict()

        # Validate passenger data
        validation_errors = validate_passenger_data(passenger_data)
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid passenger data: {'; '.join(validation_errors)}"
            )

        # Make prediction
        survived, probability = model_manager.predict(passenger_data)

        # Record metrics
        prediction_time = time.time() - start_time
        metrics_collector.record_request(prediction_time, success=True)

        # Log prediction
        logger.info("Prediction made",
                    passenger_id=passenger_data["passenger_id"],
                    survived=survived,
                    probability=probability,
                    prediction_time=prediction_time)

        return PredictionResponse(
            prediction=Prediction(
                passenger_id=passenger_data["passenger_id"],
                survived=survived,
                survival_probability=probability
            ),
            model_version=model_manager.model_version,
            prediction_time=prediction_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction failed", error=str(e), exc_info=True)
        metrics_collector.record_request(
            time.time() - start_time, success=False)
        raise HTTPException(status_code=500, detail="Prediction failed")


# Batch prediction endpoint
@app.post("/predict/batch",
          response_model=BatchPredictionResponse,
          dependencies=[Depends(validate_api_key)],
          tags=["Predictions"])
async def predict_survival_batch(request: BatchPredictionRequest):
    """Predict survival for multiple passengers."""
    start_time = time.time()

    try:
        model_manager = get_model_manager()
        metrics_collector = get_metrics_collector()

        if not model_manager.is_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Validate batch size
        if len(request.passengers) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Batch size cannot exceed 1000 passengers"
            )

        # Convert passengers to dict for processing
        passengers_data = [passenger.dict()
                           for passenger in request.passengers]

        # Validate all passenger data
        for i, passenger_data in enumerate(passengers_data):
            validation_errors = validate_passenger_data(passenger_data)
            if validation_errors:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid passenger data at index {i}: {'; '.join(validation_errors)}"
                )

        # Make predictions
        predictions = model_manager.predict_batch(passengers_data)

        # Format results
        prediction_results = []
        for passenger, (survived, probability) in zip(request.passengers, predictions):
            prediction_results.append(Prediction(
                passenger_id=passenger.passenger_id,
                survived=survived,
                survival_probability=probability
            ))

        # Record metrics
        prediction_time = time.time() - start_time
        metrics_collector.record_request(prediction_time, success=True)

        # Log batch prediction
        logger.info("Batch prediction made",
                    passenger_count=len(passengers_data),
                    prediction_time=prediction_time)

        return BatchPredictionResponse(
            predictions=prediction_results,
            model_version=model_manager.model_version,
            prediction_time=prediction_time,
            total_passengers=len(passengers_data)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch prediction failed", error=str(e), exc_info=True)
        metrics_collector.record_request(
            time.time() - start_time, success=False)
        raise HTTPException(status_code=500, detail="Batch prediction failed")


# Model information endpoint
@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get model information and feature importance."""
    model_manager = get_model_manager()

    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get feature importance
    feature_importance = model_manager.get_feature_importance()

    # Get model metrics
    model_metrics = {}
    if model_manager.model_metrics:
        model_metrics = model_manager.model_metrics.get("metrics", {})

    return ModelInfoResponse(
        model_version=model_manager.model_version,
        feature_importance=feature_importance,
        model_metrics=model_metrics,
        training_date=datetime.fromtimestamp(
            model_manager.load_time
        ).isoformat() if model_manager.load_time else None
    )


# Metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get API performance metrics."""
    metrics_collector = get_metrics_collector()
    return metrics_collector.get_metrics()


# API key generation endpoint (for testing)
@app.post("/admin/generate-api-key", tags=["Admin"])
async def generate_api_key(user_id: str):
    """Generate a new API key for testing."""
    security_manager = get_security_manager()
    api_key = security_manager.generate_api_key(user_id)

    logger.info("API key generated", user_id=user_id)

    return {"api_key": api_key}


# Example prediction endpoint
@app.get("/examples", tags=["Examples"])
async def get_examples():
    """Get example requests for testing."""
    return {
        "single_prediction": {
            "passenger": {
                "passenger_id": 1,
                "pclass": 1,
                "sex": "female",
                "age": 29.0,
                "sibsp": 0,
                "parch": 0,
                "fare": 211.3375,
                "embarked": "S"
            }
        },
        "batch_prediction": {
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
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
