from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from typing import List


class Item(BaseModel):
    data: List[List[float]]

def load_model(path='models/model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


app = FastAPI()
model = load_model()

@app.post("/predict")
def predict(item: Item):
    prediction = model.predict(item.data).tolist()
    return {
        'prediction': prediction
    }
