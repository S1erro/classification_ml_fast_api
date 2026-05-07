from sklearn.model_selection import train_test_split
from pandas import DataFrame, Series

def split_train_test(
        X: DataFrame,
        y: Series
    ) -> tuple[DataFrame, DataFrame, Series, Series]:

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        stratify=y, 
        random_state=2
    )

    return X_train, X_test, y_train, y_test

def prepare_dataframe(df: DataFrame) -> tuple[DataFrame, Series]:
    local_dataframe = df.dropna(subset=['churn'])
    
    X = local_dataframe.drop(columns=["churn"])
    y = local_dataframe["churn"]

    numeric_features = [
        "monthly_fee",
        "usage_hours",
        "support_requests",
        "account_age_months",
        "failed_payments",
    ]

    categorical_features = [
        "region",
        "device_type",
        "payment_method",
        "autopay_enabled",
    ]

    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].mean())
    X[categorical_features] = X[categorical_features].fillna("unknown")

    return X, y

def get_churn_distribution(y_test: Series, y_train: Series) -> tuple[float, float]:
    test_distribution = (y_test == 1).sum() / (y_test == 0).sum()
    train_distribution = (y_train == 1).sum() / (y_train == 0).sum()

    return test_distribution, train_distribution
