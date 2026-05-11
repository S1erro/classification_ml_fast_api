import pandas as pd

from src.models.models import TrainingConfigChurn
from src.pipelines.build_model_pypeline import build_model_pipeline
from src.types.constants.dataframe_columns import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.types.model_type import ModelType
from src.utils.prepare_data import prepare_dataframe, split_train_test
from src.utils.train_model import train_churn_model


def _make_synthetic_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "monthly_fee": [10.0, 30.0, None, 25.0, 5.0, 40.0, 9.0, 19.0, 22.0, None],
            "usage_hours": [5.0, 35.0, 20.0, None, 3.0, 45.0, 7.0, 12.0, None, 40.0],
            "support_requests": [0, 4, 1, 2, 0, 5, 1, 2, 3, 4],
            "account_age_months": [2, 24, 10, 8, 1, 40, 5, 12, 14, 30],
            "failed_payments": [0, 2, 1, 1, 0, 3, 0, 1, 2, 3],
            "region": ["eu", "us", None, "eu", "apac", "us", "eu", "apac", "us", None],
            "device_type": ["mobile", "desktop", "desktop", None, "mobile", "desktop", "mobile", "desktop", None, "mobile"],
            "payment_method": ["card", "paypal", "card", "card", None, "paypal", "card", "card", "paypal", None],
            "autopay_enabled": [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
            "churn": [0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
        }
    )


def test_prepare_dataframe_fills_missing_and_splits_features_target() -> None:
    df = _make_synthetic_df()

    X, y = prepare_dataframe(df)

    assert "churn" not in X.columns
    assert len(X) == len(y) == len(df)
    assert X[NUMERIC_COLUMNS].isna().sum().sum() == 0
    assert X[CATEGORICAL_COLUMNS].isna().sum().sum() == 0


def test_train_pipeline_on_synthetic_data_and_predict() -> None:
    df = _make_synthetic_df()
    X, y = prepare_dataframe(df)
    X_train, X_test, y_train, _ = split_train_test(X, y)

    config = TrainingConfigChurn(model_type=ModelType.LOG_REG, hyperparameters={"max_iter": 200})
    pipeline = build_model_pipeline(config, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)

    model = train_churn_model(X_train, y_train, pipeline)
    preds = model.predict(X_test)

    assert len(preds) == len(X_test)
