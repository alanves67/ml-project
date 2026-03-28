from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
import uuid
import pandas as pd

from src.utils.logger import setup_logger
from src.model.train import ModelTrainer
from src.model.predict import ModelPredictor
from src.utils.data_generator import generate_synthetic_data
from src.database.db import PredictionDatabase

logger = setup_logger("api")
app = FastAPI(title="ML Training API", version="1.0.0")

# Инициализация
trainer = ModelTrainer()
predictor = None

# Инициализируем базу данных
db = None
try:
    db = PredictionDatabase()
    logger.info("Database connection established")
except Exception as e:
    logger.warning(f"Database connection failed: {e}. Running without database.")


# Pydantic модели
class TrainRequest(BaseModel):
    n_samples: int = Field(1000, ge=100, le=10000)
    n_features: int = Field(3, ge=1, le=10)
    noise: float = Field(0.1, ge=0.0, le=1.0)


class TrainResponse(BaseModel):
    status: str
    metrics: Dict[str, float]
    coefficients: Dict[str, float]
    model_path: str
    message: str


class PredictRequest(BaseModel):
    features: List[Dict[str, float]] = Field(
        ...,
        description="Список объектов с признаками. Признаки: feature_1, feature_2, feature_3",
        example=[
            {"feature_1": 1.0, "feature_2": 2.0, "feature_3": 3.0},
            {"feature_1": -1.0, "feature_2": 0.5, "feature_3": 2.0}
        ]
    )

    @field_validator('features')
    def check_features(cls, v):
        if not v:
            raise ValueError('Features list cannot be empty')
        return v


class PredictResponse(BaseModel):
    predictions: List[float]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class PredictionHistoryResponse(BaseModel):
    id: int
    timestamp: str
    features: Dict[str, float]
    prediction: float
    user_ip: Optional[str] = None
    request_id: Optional[str] = None


class PredictionStatsResponse(BaseModel):
    total_predictions: int
    avg_prediction: Optional[float] = None
    min_prediction: Optional[float] = None
    max_prediction: Optional[float] = None
    database_connected: bool


# Эндпоинты
@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        version="1.0.0"
    )


@app.post("/train", response_model=TrainResponse, status_code=status.HTTP_201_CREATED)
async def train_model(request: TrainRequest):
    global predictor

    try:
        logger.info(f"Training request received: {request.model_dump()}")

        X, y = generate_synthetic_data(
            n_samples=request.n_samples,
            n_features=request.n_features,
            noise=request.noise
        )

        metrics = trainer.train(X, y)
        coefficients = metrics.pop('coefficients')

        model_path = trainer.save_model()
        predictor = ModelPredictor(model_path)

        return TrainResponse(
            status="success",
            metrics=metrics,
            coefficients=coefficients,
            model_path=model_path,
            message=f"Model trained successfully on {request.n_samples} samples"
        )

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest,
                  req: Request,
                  request_id: Optional[str] = None):
    global predictor

    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model not trained yet. Please train a model first using /train endpoint"
        )

    try:
        # Генерируем ID запроса
        if request_id is None:
            request_id = str(uuid.uuid4())

        client_ip = req.client.host if req.client else None

        logger.info(f"Prediction request received: {len(request.features)} samples, request_id={request_id}")

        # Делаем предсказания
        predictions = predictor.predict(request.features)

        # Сохраняем в базу данных
        if db is not None:
            for features, prediction in zip(request.features, predictions):
                try:
                    db.log_prediction(
                        features=features,
                        prediction=float(prediction),
                        user_ip=client_ip,
                        request_id=request_id
                    )
                except Exception as e:
                    logger.error(f"Failed to save prediction to DB: {e}")

        return PredictResponse(
            predictions=predictions.tolist(),
            count=len(predictions)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/predictions/history", response_model=List[PredictionHistoryResponse])
async def get_prediction_history(limit: int = 100):
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )

    try:
        history = db.get_recent_predictions(limit=limit)
        return history
    except Exception as e:
        logger.error(f"Failed to get prediction history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get history: {str(e)}"
        )


@app.get("/predictions/stats", response_model=PredictionStatsResponse)
async def get_prediction_stats():
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )

    try:
        stats = db.get_prediction_stats()
        stats['database_connected'] = True
        return stats
    except Exception as e:
        logger.error(f"Failed to get prediction stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded"
        )

    return {
        "model_type": type(predictor.model).__name__,
        "model_loaded": True,
        "model_path": str(predictor.model_path)
    }