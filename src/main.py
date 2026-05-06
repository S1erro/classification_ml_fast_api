from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "ml churn service is running"}

if __name__ == "__main__":
    pass