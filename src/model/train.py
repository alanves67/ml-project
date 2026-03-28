import joblib
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from typing import Tuple, Dict

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelTrainer:
    """Класс для обучения модели"""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model = LinearRegression()
        logger.info(f"ModelTrainer initialized with model_dir={model_dir}")

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Обучает модель на данных"""
        logger.info(f"Starting training with {len(X)} samples, {X.shape[1]} features")

        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        logger.info(f"Training completed - MSE: {mse:.4f}, R2: {r2:.4f}")

        return {
            'mse': mse,
            'r2': r2,
            'intercept': self.model.intercept_,
            'coefficients': dict(zip(X.columns, self.model.coef_))
        }

    def save_model(self, filename: str = "linear_model.joblib") -> str:
        """Сохраняет модель в файл"""
        model_path = self.model_dir / filename
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        return str(model_path)

    def load_model(self, filename: str = "linear_model.joblib") -> None:
        """Загружает модель из файла"""
        model_path = self.model_dir / filename
        if not model_path.exists():
            logger.error(f"Model file {model_path} not found")
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")