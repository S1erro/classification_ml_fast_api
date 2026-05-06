from typing import List

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