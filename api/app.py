from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_one

app = FastAPI(title="Smart Support Triage API")

class Input(BaseModel):
    text: str

@app.post("/predict")
def _predict(x: Input):
    return predict_one(x.text)
