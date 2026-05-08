from dataclasses import dataclass

from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import joblib

from ..models import TrainModelMetrics

@dataclass
class SavedModel:
    model: Pipeline
    timestamp: float
    metrics: TrainModelMetrics

def save_model(model: SavedModel, model_name: str) -> None:
    joblib.dump(model, model_name)

def load_model(model_name: str) -> SavedModel:
    model = joblib.load(model_name)
    return model