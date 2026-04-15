"""
API FastAPI - Wine Quality Prediction Service
Endpoints:
  GET  /health   → statut du service
  POST /predict  → prédiction qualité d'un vin
"""

import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Chargement des artefacts ---
MODEL_PATH = Path("artifacts/model.pkl")
SCALER_PATH = Path("artifacts/scaler.pkl")
FEATURES_PATH = Path("artifacts/feature_names.json")

# Si les artefacts sont absents, on entraîne à la volée (pratique en dev)
if not MODEL_PATH.exists():
    import train as _train
    _train.main()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH) as f:
    FEATURE_NAMES = json.load(f)

# --- App ---
app = FastAPI(
    title="Wine Quality API",
    description="Prédit si un vin est de bonne qualité (1) ou non (0)",
    version="1.0.0",
)


# --- Schémas Pydantic ---
class WineFeatures(BaseModel):
    type: float           # 0=red, 1=white
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": 1,
                "fixed_acidity": 7.0,
                "volatile_acidity": 0.27,
                "citric_acid": 0.36,
                "residual_sugar": 20.7,
                "chlorides": 0.045,
                "free_sulfur_dioxide": 45.0,
                "total_sulfur_dioxide": 170.0,
                "density": 1.001,
                "pH": 3.0,
                "sulphates": 0.45,
                "alcohol": 8.8,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_good: float


# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForestClassifier", "dataset": "Wine Quality"}


@app.post("/predict", response_model=PredictionResponse)
def predict(wine: WineFeatures):
    try:
        # Mise en forme dans le bon ordre
        features = [
            wine.type,
            wine.fixed_acidity,
            wine.volatile_acidity,
            wine.citric_acid,
            wine.residual_sugar,
            wine.chlorides,
            wine.free_sulfur_dioxide,
            wine.total_sulfur_dioxide,
            wine.density,
            wine.pH,
            wine.sulphates,
            wine.alcohol,
        ]
        X = np.array([features])
        X_scaled = scaler.transform(X)

        pred = int(model.predict(X_scaled)[0])
        proba = float(model.predict_proba(X_scaled)[0][1])

        return PredictionResponse(
            prediction=pred,
            label="Bon" if pred == 1 else "Mauvais",
            probability_good=round(proba, 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
