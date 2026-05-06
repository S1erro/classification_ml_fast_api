from typing import List

import pandas as pd

from .models import DatasetRowChurn

class UseCsvData:    
    def __init__(self) -> None:
        self.df = pd.DataFrame()

    def read_csv(self, path: str) -> None:
        self.df = pd.read_csv(path)

    def transform_row_to_object(self, start_row: int, end_row: int) -> List[DatasetRowChurn]:
        transformed_rows: List[DatasetRowChurn] = []
        rows_in_range = self.df.iloc[start_row : end_row + 1]

        transformed_rows = [
            DatasetRowChurn(
                monthly_fee=row.monthly_fee,
                usage_hours=row.usage_hours,
                support_requests=row.support_requests,
                account_age_months=row.account_age_months,
                failed_payments=row.failed_payments,
                region=row.region,
                device_type=row.device_type,
                payment_method=row.payment_method,
                autopay_enabled=row.autopay_enabled,
                churn=row.churn
            )
            for row in rows_in_range
        ]

        return transformed_rows