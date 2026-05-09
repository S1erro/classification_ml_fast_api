from dataclasses import dataclass

from sklearn.pipeline import Pipeline

from ..models.models import TrainModelMetrics, TrainingConfigChurn


@dataclass
class SavedModel:
    model: Pipeline
    timestamp: float
    metrics: TrainModelMetrics
    training_config: TrainingConfigChurn