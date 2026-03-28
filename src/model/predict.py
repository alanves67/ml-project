import joblib
import pandas as pd
import numpy as np
from typing import Union, List, Dict
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelPredictor:
    """Класс для предсказаний"""

    def __init__(self, model_path: str = "models/linear_model.joblib"):
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        """Загружает модель из файла"""
        if not self.model_path.exists():
            logger.error(f"Model not found at {self.model_path}")
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        logger.info(f"Model loaded from {self.model_path}")

    def predict(self, features: Union[pd.DataFrame, List[Dict], np.ndarray]) -> np.ndarray:
        """Делает предсказания"""
        if isinstance(features, list):
            features = pd.DataFrame(features)
        elif isinstance(features, np.ndarray):
            features = pd.DataFrame(features, columns=[f'feature_{i + 1}' for i in range(features.shape[1])])

        logger.debug(f"Predicting for {len(features)} samples")
        predictions = self.model.predict(features)

        return predictions

    def predict_single(self, features: Dict[str, float]) -> float:
        """Делает предсказание для одного объекта"""
        df = pd.DataFrame([features])
        prediction = self.predict(df)
        return float(prediction[0])