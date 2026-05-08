from typing import List

from fastapi import FastAPI
from sklearn.metrics import accuracy_score, f1_score

from .constants.constants import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from .pipelines.logistic_regression import build_logistic_regression_pipeline
from .utils.is_dataframe_empty import is_dataframe_empty
from .utils.prepare_data import get_churn_distribution, prepare_dataframe, split_train_test
from .utils.read_csv import UseCsvData
from .utils.train_model import train_churn_model
from .models import DatasetInfo, DatasetRowChurn, FeatureVectorChurn, SplitInfo, TrainModel


app = FastAPI()

csv_data = UseCsvData()
csv_data.read_csv("data/churn_dataset.csv")

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

@app.post("/model/train")
async def train_model() -> TrainModel:
    is_dataframe_empty(csv_data.df)

    X, y = prepare_dataframe(csv_data.df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    pipeline = build_logistic_regression_pipeline(NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)

    model = train_churn_model(X_train, y_train, pipeline)
    predictions = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions))

    return TrainModel(
        accuracy=accuracy,
        f1_score=f1
    )

@app.post("/predict")
async def predict(vector: FeatureVectorChurn) -> FeatureVectorChurn:
    return vector

if __name__ == "__main__":
    pass