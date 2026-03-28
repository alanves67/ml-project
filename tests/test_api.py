import pytest
from fastapi.testclient import TestClient
from src.api.app import app


class TestAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_train_and_predict(self, client):
        # Train
        train_response = client.post(
            "/train",
            json={"n_samples": 100, "n_features": 3, "noise": 0.1}
        )
        assert train_response.status_code == 201

        # Predict
        predict_response = client.post(
            "/predict",
            json={"features": [{"feature_1": 1.0, "feature_2": 2.0, "feature_3": 3.0}]}
        )
        assert predict_response.status_code == 200
        data = predict_response.json()
        assert len(data["predictions"]) == 1 