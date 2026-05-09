import time
import os
import pandas as pd
from typing import Any, List, Union

from fastapi import FastAPI, HTTPException
from sklearn.metrics import accuracy_score, f1_score

from .types.model_type import ModelType
from .types.constants.dataframe_columns import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from .pipelines.build_model_pypeline import build_model_pipeline
from .utils.is_dataframe_empty import is_dataframe_empty
from .utils.prepare_data import get_churn_distribution, prepare_dataframe, split_train_test
from .utils.read_csv import UseCsvData
from .utils.train_model import train_churn_model
from .utils.save_load_models import SavedModel, load_model, save_model
from .models.models import DatasetInfo, DatasetRowChurn, FeatureVectorChurn, ModelStatus, PredictionResponseChurn, SplitInfo, TrainModelMetrics, TrainingConfigChurn


app = FastAPI()

csv_data = UseCsvData()
csv_data.read_csv("data/churn_dataset.csv")

saved_model: Union[SavedModel, None] = None

try:
    saved_model = load_model("src/cached_models/linear_regression.joblib")
except:
    saved_model = None

@app.get("/")
async def root():
    return {"message": "ml churn service is running"}

@app.get("/dataset/preview")
async def get_dataset_preview(rows_count: int) -> List[DatasetRowChurn]:
    is_dataframe_empty(csv_data.df)

    transformed_row = csv_data.transform_row_to_object(0, rows_count)
    return transformed_row

@app.get("/dataset/info")
async def get_dataset_info() -> DatasetInfo:
    is_dataframe_empty(csv_data.df)

    shape = csv_data.df.shape
    columns = csv_data.df.columns.tolist()

    dataset_info: DatasetInfo = DatasetInfo(
        rows_count=shape[0],
        columns_count=shape[1],
        titles=columns,
        churn_distribution={'temporary plug': 0.29}
    )
    return dataset_info

@app.get("/dataset/split-info")
async def get_dataset_split_info() -> SplitInfo:
    is_dataframe_empty(csv_data.df)

    X, y = prepare_dataframe(csv_data.df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    test_distribution, train_distribution = get_churn_distribution(y_test, y_train)

    return SplitInfo(
        train_shape=X_train.shape,
        test_shape=X_test.shape,
        test_distribution=test_distribution,
        train_distribution=train_distribution
    )

@app.get("/model/status")
async def get_model_status() -> ModelStatus:
    """
    Get status of the last trained model
    """
    global saved_model

    is_trained = False
    timestamp = None
    metrics = None
    training_config = None

    if saved_model:
        is_trained = True
        timestamp = saved_model.timestamp
        metrics = saved_model.metrics
        training_config = saved_model.training_config

    return ModelStatus(
        is_trained=is_trained,
        timestamp=timestamp,
        metrics=metrics,
        training_config=training_config
    )

@app.get("/model/schema")
async def get_required_features() -> dict[str, Any]:
    return FeatureVectorChurn.model_json_schema()

@app.post("/model/train")
async def train_model(training_config: TrainingConfigChurn) -> TrainModelMetrics:
    global saved_model

    is_dataframe_empty(csv_data.df)

    X, y = prepare_dataframe(csv_data.df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    pipeline = build_model_pipeline(training_config, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)

    model = train_churn_model(X_train, y_train, pipeline)
    predictions = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions))

    metrics = TrainModelMetrics(
        accuracy=accuracy,
        f1_score=f1
    )

    model_to_save: SavedModel = SavedModel(
        model=model,
        timestamp=time.time(),
        metrics=metrics,
        training_config=training_config
    )

    os.makedirs("src/cached_models", exist_ok=True)
    if training_config.model_type == ModelType.LOG_REG:
        save_model(model_to_save, "src/cached_models/linear_regression.joblib")
    elif training_config.model_type == ModelType.RAND_FOREST:
        save_model(model_to_save, "src/cached_models/random_forest.joblib")

    saved_model = model_to_save

    return metrics

@app.post("/predict")
async def predict(vector: FeatureVectorChurn) -> PredictionResponseChurn:
    """
    Predict by using the last trained model
    """
    global saved_model

    expected_cols = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    df = pd.DataFrame(data=[vector.model_dump()]).reindex(columns=expected_cols)

    prediction = None
    prediction_proba = None

    if saved_model != None:
        prediction = saved_model.model.predict(df)
        prediction_proba = saved_model.model.predict_proba(df)
    else:
        raise HTTPException(status_code=500, detail="Model is not trained")


    return PredictionResponseChurn(
        predicted_class=int(prediction[0]),
        classes_probabilities=prediction_proba[0].tolist()
    )

if __name__ == "__main__":
    pass