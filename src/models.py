from typing import List, Optional

from pydantic import BaseModel

class FeatureVectorChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int

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
    churn_distribution: dict[str, float]

class SplitInfo(BaseModel): # модель для POST /dataset/split-info
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]
    test_distribution: dict
    train_distribution: dict

class TrainModelMetrics(BaseModel): # модель для POST /model/train
    accuracy: float
    f1_score: float

class ModelStatus(BaseModel): # модель для GET /model/status
    is_trained: bool
    timestamp: Optional[float]
    metrics: Optional[TrainModelMetrics]