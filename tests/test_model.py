import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.model.train import ModelTrainer
from src.model.predict import ModelPredictor
from src.utils.data_generator import generate_synthetic_data


class TestModelTrainer:
    def test_train(self):
        trainer = ModelTrainer(model_dir=tempfile.mkdtemp())
        X, y = generate_synthetic_data(n_samples=100)
        metrics = trainer.train(X, y)

        assert 'mse' in metrics
        assert 'r2' in metrics
        assert metrics['mse'] >= 0

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(model_dir=tmpdir)
            X, y = generate_synthetic_data(n_samples=100)
            trainer.train(X, y)
            model_path = trainer.save_model("test_model.joblib")

            assert os.path.exists(model_path)

            predictor = ModelPredictor(model_path)
            assert predictor.model is not None