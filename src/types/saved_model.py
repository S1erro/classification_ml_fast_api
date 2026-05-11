from dataclasses import dataclass
from enum import Enum
from typing import Any

from sklearn.pipeline import Pipeline

from ..types.model_type import ModelType
from ..models.models import TrainModelMetrics, TrainingConfigChurn

def _serialize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    return value

@dataclass
class BaseSavedModel:
    timestamp: float
    metrics: TrainModelMetrics
    training_config: TrainingConfigChurn

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "metrics": _serialize(self.metrics),
            "training_config": _serialize(self.training_config),
        }

@dataclass
class SavedModel(BaseSavedModel):
    model: Pipeline

@dataclass 
class SavedModelToHistory(BaseSavedModel):
    model_type: ModelType

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["model_type"] = _serialize(self.model_type)
        return data