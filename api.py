from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from trainer import Trainer
from typing import Dict, List

app = FastAPI()
trainer = Trainer()

class TrainingData(BaseModel):
    texts: List[str]
    labels: List[str]

class TestingData(BaseModel):
    texts: List[str]

class QueryObject(BaseModel):
    text: str

class StatusObject(BaseModel):
    status: str
    timestamp: str
    classes: List[str]
    evaluation: Dict

class PredictionObject(BaseModel):
    text: str
    predictions: Dict

class PredictionsObject(BaseModel):
    predictions: List[PredictionObject]

@app.get("/")
def home():
    return({"message": "API is working properly."})

@app.get("/status", summary="Get syster status")
def get_status():
    status = trainer.get_status()
    return StatusObject(**status)

@app.post("/train", summary="Train a new model")
def train(training_data:TrainingData):
    try:
        trainer.train(training_data.texts, training_data.labels)
        status = trainer.get_status()
        return StatusObject(**status)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict", summary="Predict the probability of each label for a single input")
def predict(query_text: QueryObject):
    try:
        prediction = trainer.predict([query_text.text])[0]
        return PredictionObject(**prediction)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict-batch", summary="Predict the probability of each label for a batch of inputs")
def predict_batch(testing_data:TestingData):
    try:
        predictions = trainer.predict(testing_data.texts)
        return PredictionsObject(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))