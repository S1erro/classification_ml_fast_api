from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..types.constants.response_error_types import ResponseErrorTypes
from ..types.model_type import ModelType

class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monthly_fee: float = Field(..., json_schema_extra={"example": 9.99})
    usage_hours: float = Field(..., json_schema_extra={"example": 27.92})
    support_requests: int = Field(..., json_schema_extra={"example": 1})
    account_age_months: int = Field(..., json_schema_extra={"example": 14})
    failed_payments: int = Field(..., json_schema_extra={"example": 1})
    region: str = Field(..., json_schema_extra={"example": "america"})
    device_type: str = Field(..., json_schema_extra={"example": "desktop"})
    payment_method: str = Field(..., json_schema_extra={"example": "card"})
    autopay_enabled: int = Field(..., json_schema_extra={"example": 1})

class DatasetRowChurn(BaseModel): # строка тренировочного датасета
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int
    churn:int

class DatasetInfo(BaseModel): # модель для GET /dataset/info
    rows_count: int
    columns_count: int
    titles: List[str]
    churn_distribution: Dict[str, float]

class SplitInfo(BaseModel): # модель для POST /dataset/split-info
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]
    test_distribution: Dict
    train_distribution: Dict

class TrainModelMetrics(BaseModel): # модель для POST /model/train
    accuracy: float
    f1_score: float
    roc_auc: float

class PredictionResponseChurn(BaseModel): # модель для POST /predict
    predicted_class: int = Field(..., json_schema_extra={"example": 0})
    classes_probabilities: tuple[float, float] = Field(..., json_schema_extra={"example": [0.6781, 0.3219]})

class TrainingConfigChurn(BaseModel):
    model_type: ModelType = Field(..., json_schema_extra={"example": "logreg"})
    hyperparameters: Dict = Field(..., json_schema_extra={"example": {"max_iter": 1000}})

class ModelStatus(BaseModel): # модель для GET /model/status
    is_trained: bool
    timestamp: Optional[float]
    metrics: Optional[TrainModelMetrics]
    training_config: Optional[TrainingConfigChurn]

class ErrorResponse(BaseModel): # модель HTTP ошибок
    code: ResponseErrorTypes
    message: str
    details: Optional[Dict[str, Any]] = None