import os
from typing import Any, Dict

from fastapi import FastAPI, Body, HTTPException

# Import the new prediction interface
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'RealTimeCarPricePrediction'))

from src.price_model.model_comparison import ModelComparison
import joblib
import pandas as pd

ARTIFACT_PATH = os.getenv("ARTIFACT_PATH", "RealTimeCarPricePrediction/artifacts/final_model.joblib")

app = FastAPI(title="Car Price API", version="0.1.0")

# Global model storage
_PIPELINE = None

def load_model(path: str = ARTIFACT_PATH):
    """Load the trained model pipeline."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = joblib.load(path)
    return _PIPELINE

def predict_one(payload: Dict[str, Any], artifact_path: str = ARTIFACT_PATH) -> float:
    """Make a prediction for a single record."""
    pipeline = load_model(artifact_path)
    X = pd.DataFrame([payload])
    yhat = float(pipeline.predict(X)[0])
    return yhat

@app.on_event("startup")
def _startup():
    # Load once so readiness reflects a loaded pipeline
    try:
        load_model(ARTIFACT_PATH)
        app.state.ready = True
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        app.state.ready = False

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    return {"ready": bool(getattr(app.state, "ready", False))}

@app.post("/predict")
def predict(payload: Dict[str, Any] = Body(...)):
    """
    Accepts a single-record JSON payload (feature_name: value) and returns a price prediction.
    """
    try:
        yhat = predict_one(payload, artifact_path=ARTIFACT_PATH)
        return {"prediction": yhat}
    except Exception as e:
        # Surface a clean 400 on bad inputs (schema/type issues etc.)
        raise HTTPException(status_code=400, detail=str(e))
