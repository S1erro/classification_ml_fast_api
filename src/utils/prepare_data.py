from sklearn.model_selection import train_test_split
from pandas import DataFrame, Series

from ..types.constants.dataframe_columns import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

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

    X[NUMERIC_COLUMNS] = X[NUMERIC_COLUMNS].fillna(X[NUMERIC_COLUMNS].mean())
    X[CATEGORICAL_COLUMNS] = X[CATEGORICAL_COLUMNS].fillna("unknown")

    return X, y

def get_churn_distribution(y_test: Series, y_train: Series) -> tuple[dict, dict]:
    test_distribution = y_test.value_counts(normalize=True).to_dict()
    train_distribution = y_train.value_counts(normalize=True).to_dict()

    return test_distribution, train_distribution
