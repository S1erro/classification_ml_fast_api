from typing import List, cast

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
                monthly_fee=float(cast(float, row.monthly_fee)),
                usage_hours=float(cast(float, row.usage_hours)),
                support_requests=int(cast(int, row.support_requests)),
                account_age_months=int(cast(int, row.account_age_months)),
                failed_payments=int(cast(int, row.failed_payments)),
                region=str(cast(str, row.region)),
                device_type=str(cast(str, row.device_type)),
                payment_method=str(cast(str, row.payment_method)),
                autopay_enabled=int(cast(int, row.autopay_enabled)),
                churn=int(cast(int, row.churn)),
            )
            for row in rows_in_range.itertuples(index=False)
        ]

        return transformed_rows