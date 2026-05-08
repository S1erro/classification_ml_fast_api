import time
import os
from typing import List, Union

from fastapi import FastAPI
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from .constants.constants import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from .pipelines.logistic_regression import build_logistic_regression_pipeline
from .utils.is_dataframe_empty import is_dataframe_empty
from .utils.prepare_data import get_churn_distribution, prepare_dataframe, split_train_test
from .utils.read_csv import UseCsvData
from .utils.train_model import train_churn_model
from .utils.save_load_models import SavedModel, load_model, save_model
from .models import DatasetInfo, DatasetRowChurn, FeatureVectorChurn, ModelStatus, SplitInfo, TrainModelMetrics


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
    is_trained = False
    timestamp = None
    metrics = None 

    if saved_model:
        is_trained = True
        timestamp = saved_model.timestamp
        metrics = saved_model.metrics

    return ModelStatus(
        is_trained=is_trained,
        timestamp=timestamp,
        metrics=metrics
    )

@app.post("/model/train")
async def train_model() -> TrainModelMetrics:
    global saved_model

    is_dataframe_empty(csv_data.df)

    X, y = prepare_dataframe(csv_data.df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    pipeline = build_logistic_regression_pipeline(NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)

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
        metrics=metrics
    )

    os.makedirs("src/cached_models", exist_ok=True)
    save_model(model_to_save, "src/cached_models/linear_regression.joblib")
    saved_model = model_to_save

    return metrics

@app.post("/predict")
async def predict(vector: FeatureVectorChurn) -> FeatureVectorChurn:
    return vector

if __name__ == "__main__":
    pass