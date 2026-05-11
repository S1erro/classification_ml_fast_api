from pathlib import Path

from fastapi.testclient import TestClient

import src.main as app_module
from src.main import app


def _valid_train_payload() -> dict:
    return {
        "model_type": "logreg",
        "hyperparameters": {"max_iter": 200},
    }


def _valid_predict_payload() -> dict:
    return {
        "monthly_fee": 25.0,
        "usage_hours": 40.0,
        "support_requests": 2,
        "account_age_months": 15,
        "failed_payments": 1,
        "region": "us",
        "device_type": "desktop",
        "payment_method": "card",
        "autopay_enabled": 1,
    }


def test_full_churn_pipeline_train_status_predict(tmp_path: Path) -> None:
    original_cached_path = app_module.cached_models_path
    original_history_path = app_module.train_history_path
    original_saved_model = app_module.saved_model

    app_module.cached_models_path = str(tmp_path / "cached_models")
    app_module.train_history_path = str(tmp_path / "train_history")
    app_module.saved_model = None

    Path(app_module.cached_models_path).mkdir(parents=True, exist_ok=True)
    Path(app_module.train_history_path).mkdir(parents=True, exist_ok=True)

    client = TestClient(app)

    try:
        health_before_train_response = client.get("/health")
        assert health_before_train_response.status_code == 200
        assert health_before_train_response.json() == {
            "model_available": False,
            "dataset_loaded": True,
        }

        train_response = client.post("/model/train", json=_valid_train_payload())
        assert train_response.status_code == 200
        train_json = train_response.json()
        assert "accuracy" in train_json
        assert "f1_score" in train_json
        assert "roc_auc" in train_json

        status_response = client.get("/model/status")
        assert status_response.status_code == 200
        status_json = status_response.json()
        assert status_json["is_trained"] is True
        assert status_json["metrics"] is not None

        health_after_train_response = client.get("/health")
        assert health_after_train_response.status_code == 200
        assert health_after_train_response.json() == {
            "model_available": True,
            "dataset_loaded": True,
        }

        predict_response = client.post("/predict", json=_valid_predict_payload())
        assert predict_response.status_code == 200
        predict_json = predict_response.json()
        assert "predicted_class" in predict_json
        assert len(predict_json["classes_probabilities"]) == 2
    finally:
        app_module.cached_models_path = original_cached_path
        app_module.train_history_path = original_history_path
        app_module.saved_model = original_saved_model


def test_predict_without_trained_model_returns_error() -> None:
    original_saved_model = app_module.saved_model
    app_module.saved_model = None

    client = TestClient(app)

    try:
        response = client.post("/predict", json=_valid_predict_payload())
        assert response.status_code == 500
        body = response.json()
        assert body["code"] == "MODEL_NOT_TRAINED"
    finally:
        app_module.saved_model = original_saved_model
