import uvicorn
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="UFO Sighting Predictor API")

model = joblib.load('ufo_model.pkl')
le = joblib.load('ufo_label_encoder.pkl')

country_names = {
    'us': 'United States', 'ca': 'Canada', 'gb': 'United Kingdom',
    'au': 'Australia', 'de': 'Germany', 'fr': 'France', 'other': 'Other',
}

class Sighting(BaseModel):
    latitude: float
    longitude: float
    duration_seconds: int
    hour: int
    month: int
    day_of_week: int

@app.get("/")
def index():
    return {"ok": True, "message": "UFO Sighting Predictor API"}

@app.post("/predict")
def predict(sighting: Sighting):
    input_df = pd.DataFrame([sighting.model_dump()])
    pred_encoded = model.predict(input_df)[0]
    pred_country = le.inverse_transform([pred_encoded])[0]
    probs = model.predict_proba(input_df)[0]

    return {
        "predicted_country": pred_country,
        "predicted_country_name": country_names.get(pred_country, pred_country),
        "probabilities": {
            country_names.get(c, c): float(p)
            for c, p in zip(le.classes_, probs)
        },
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
