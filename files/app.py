from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from typing import List
import os

class Item(BaseModel):
    data: List[List[float]]

def load_model(path='models/model.pkl'):
    base_path = os.getcwd()
    path = os.path.join(base_path, path)
    with open(path, 'rb') as f:
        return pickle.load(f)


app = FastAPI()
model = load_model()

@app.post("/predict")
def predict(item: Item):
    prediction = model.predict(item.data).tolist()
    print(f"model: {model.__class__.__name__}\n"
          f"in_data: {item}\n"
          f"prediction: {prediction}\n")
    return {
        'prediction': prediction
    }
