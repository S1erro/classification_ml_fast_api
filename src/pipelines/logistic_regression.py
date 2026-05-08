from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_logistic_regression_pipeline(
        numeric_columns: List[str],
        categorical_columns: List[str]
    ) -> Pipeline:

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns)
        ]
    )
    

    model = Pipeline(
        steps=[
            ("scaler", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )
    
    
    return model