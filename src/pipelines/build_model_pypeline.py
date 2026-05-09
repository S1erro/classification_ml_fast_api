from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..types.model_type import ModelType
from ..models.models import TrainingConfigChurn

def build_model_pipeline(
        training_config: TrainingConfigChurn,
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

    classifier = None
    if training_config.model_type == ModelType.LOG_REG:
        classifier = LogisticRegression(**training_config.hyperparameters)
    elif training_config.model_type == ModelType.RAND_FOREST:
        classifier = RandomForestClassifier(**training_config.hyperparameters)

    if (classifier == None):
        raise ValueError("Incorrect model type")

    model = Pipeline(
        steps=[
            ("scaler", preprocessing),
            ("classifier", classifier)
        ]
    )
    
    
    return model