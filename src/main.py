from fastapi import FastAPI

from .models import FeatureVectorChurn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "ml churn service is running"}

@app.post("/predict")
async def predict(vector: FeatureVectorChurn) -> FeatureVectorChurn:
    return vector

if __name__ == "__main__":
    pass