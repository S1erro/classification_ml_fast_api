from fastapi import FastAPI

from utils.is_dataframe_empty import is_dataframe_empty
from .models import DatasetInfo, FeatureVectorChurn
from .read_csv import UseCsvData

app = FastAPI()

csv_data = UseCsvData()
csv_data.read_csv("data/churn_dataset.csv")

@app.get("/")
async def root():
    return {"message": "ml churn service is running"}

@app.get("/dataset/preview")
async def get_dataset_preview(rows_count: int) -> str:
    is_dataframe_empty(csv_data.df)
    return csv_data.df.head(rows_count).to_json()

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

@app.post("/predict")
async def predict(vector: FeatureVectorChurn) -> FeatureVectorChurn:
    return vector

if __name__ == "__main__":
    pass