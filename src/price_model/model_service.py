"""
FastAPI service for car price prediction model serving.
Consolidated version with improved features and proper inverse transformations.
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime
import json
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.price_model.preprocessor import Preprocessor, DomainConfig


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    make_model__te: float = Field(..., description="Make/model target encoding")
    mechanicalGrade_ordinal: int = Field(..., description="Mechanical grade (0-4)")
    kilometers: float = Field(..., description="Total kilometers")
    vehicle_age_years: int = Field(..., description="Vehicle age in years")
    cubeCapacity: float = Field(..., description="Engine capacity")
    fuel__Diesel: int = Field(..., description="Is diesel (0/1)")
    years_since_intro_at_sale: float = Field(..., description="Years since introduction")
    aestheticGrade_ordinal: int = Field(..., description="Aesthetic grade (0-4)")
    transmission__Manual: int = Field(..., description="Is manual transmission (0/1)")
    cylinder: float = Field(..., description="Number of cylinders")
    make_MERCEDES_BENZ: int = Field(..., alias="make__MERCEDES-BENZ", description="Is Mercedes-Benz (0/1)")
    make_VOLVO: int = Field(..., alias="make__VOLVO", description="Is Volvo (0/1)")
    make_AUDI: int = Field(..., alias="make__AUDI", description="Is Audi (0/1)")
    make_VOLKSWAGEN: int = Field(..., alias="make__VOLKSWAGEN", description="Is Volkswagen (0/1)")
    colour_White: int = Field(..., alias="colour__White", description="Is white color (0/1)")
    sale_year: int = Field(..., description="Sale year")
    make_TOYOTA: int = Field(..., alias="make__TOYOTA", description="Is Toyota (0/1)")
    colour_Black: int = Field(..., alias="colour__Black", description="Is black color (0/1)")
    colour_Green: int = Field(..., alias="colour__Green", description="Is green color (0/1)")
    colour_Blue: int = Field(..., alias="colour__Blue", description="Is blue color (0/1)")
    make_SEAT: int = Field(..., alias="make__SEAT", description="Is SEAT (0/1)")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: float = Field(..., description="Predicted price in real dollars")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[float] = Field(..., description="List of predicted prices in real dollars")
    count: int = Field(..., description="Number of predictions")
    timestamp: str = Field(..., description="Prediction timestamp")


class ModelService:
    """Model service for car price prediction with inverse transformations."""
    
    def __init__(self, model_path: str = None, 
                 preprocessor_path: str = None,
                 transformers_path: str = None):
        self.model_path = model_path or os.getenv("MODEL_PATH", "artifacts/improved_final_model.joblib")
        self.preprocessor_path = preprocessor_path or os.getenv("PREPROCESSOR_PATH", "artifacts/improved_preprocessor.joblib")
        self.transformers_path = transformers_path or os.getenv("TRANSFORMERS_PATH", "artifacts/transformers.joblib")
        self.model = None
        self.preprocessor = None
        self.transformers = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model, preprocessor, and transformers."""
        try:
            # Load model
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"ModelService: Model loaded from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load preprocessor
            if os.path.exists(self.preprocessor_path):
                self.preprocessor = joblib.load(self.preprocessor_path)
                print(f"ModelService: Preprocessor loaded from {self.preprocessor_path}")
            else:
                print("ModelService: Preprocessor not found, using basic preprocessing")
                self.preprocessor = None
            
            # Load transformers
            if os.path.exists(self.transformers_path):
                self.transformers = joblib.load(self.transformers_path)
                print(f"ModelService: Transformers loaded from {self.transformers_path}")
            else:
                print("ModelService: Transformers not found, no inverse transformation will be applied")
                self.transformers = None
            
            # Get feature names from model
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
            elif hasattr(self.model, 'feature_importances_'):
                self.feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
            else:
                print("ModelService: Could not determine feature names")
                
        except Exception as e:
            print(f"ModelService: Error loading model: {e}")
            raise
    
    def preprocess_single(self, data: Dict[str, Any]) -> np.ndarray:
        """Preprocess a single prediction request."""
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Add saleDate if not present (required for preprocessing)
        if 'saleDate' not in df.columns:
            df['saleDate'] = pd.Timestamp.now()
        
        # Apply the full preprocessing pipeline
        preprocessor = Preprocessor(DomainConfig())
        df_processed = preprocessor.fit_transform(df)
        
        # Apply multicollinearity pruning if available
        if self.preprocessor is not None:
            # The preprocessor is an AdvancedFeatureSelector, not a standard sklearn transformer
            # We need to select the same features that were used during training
            if hasattr(self.preprocessor, 'selected_features_'):
                df_processed = df_processed[self.preprocessor.selected_features_]
        
        # Ensure correct feature order
        if self.feature_names is not None:
            # Add missing features with default values
            for feature in self.feature_names:
                if feature not in df_processed.columns:
                    df_processed[feature] = 0.0
            
            # Reorder columns to match training
            df_processed = df_processed[self.feature_names]
        
        return df_processed.values
    
    def preprocess_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess a batch of data."""
        # Add saleDate if not present (required for preprocessing)
        if 'saleDate' not in df.columns:
            df['saleDate'] = pd.Timestamp.now()
        
        # Apply the full preprocessing pipeline
        preprocessor = Preprocessor(DomainConfig())
        df_processed = preprocessor.fit_transform(df)
        
        # Apply multicollinearity pruning if available
        if self.preprocessor is not None:
            # The preprocessor is an AdvancedFeatureSelector, not a standard sklearn transformer
            # We need to select the same features that were used during training
            if hasattr(self.preprocessor, 'selected_features_'):
                df_processed = df_processed[self.preprocessor.selected_features_]
        
        # Ensure correct feature order
        if self.feature_names is not None:
            # Add missing features with default values
            for feature in self.feature_names:
                if feature not in df_processed.columns:
                    df_processed[feature] = 0.0
            
            # Reorder columns to match training
            df_processed = df_processed[self.feature_names]
        
        return df_processed.values
    
    def predict_single(self, data: Dict[str, Any]) -> float:
        """Make a single prediction and return inverse transformed result."""
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            start_time = time.time()
            
            # Preprocess data
            X = self.preprocess_single(data)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Apply inverse transformation if target was log-transformed
            if self.transformers and self.transformers.get('target_log_transformed', False):
                try:
                    prediction = self.transformers['target_transformer'].inverse_transform(
                        np.array([[prediction]])
                    )[0][0]
                    print("ModelService: Applied inverse transformation to prediction")
                except Exception as e:
                    print(f"Error during inverse transformation: {e}")
            
            # Record metrics
            prediction_time = time.time() - start_time
            PREDICTION_DURATION.observe(prediction_time)
            PREDICTION_COUNT.inc()
            PREDICTION_VALUE.observe(float(prediction))
            
            return float(prediction)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
    
    def predict_batch(self, df: pd.DataFrame) -> List[float]:
        """Make batch predictions and return inverse transformed results."""
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            # Preprocess data
            X = self.preprocess_batch(df)
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Apply inverse transformation to predictions if target was log-transformed
            if self.transformers and self.transformers.get('target_log_transformed', False):
                try:
                    predictions = self.transformers['target_transformer'].inverse_transform(
                        predictions.reshape(-1, 1)
                    ).flatten()
                    print("ModelService: Applied inverse transformation to predictions")
                except Exception as e:
                    print(f"Error during inverse transformation: {e}")
            
            return [float(p) for p in predictions]
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made')
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Prediction duration')
PREDICTION_VALUE = Histogram('prediction_value', 'Predicted car prices')

# Initialize FastAPI app
app = FastAPI(
    title="Car Price Prediction API",
    description="API for predicting car prices using machine learning",
    version="1.0.0"
)

# Initialize model service
model_service = ModelService()

# Middleware for metrics
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    REQUEST_DURATION.observe(process_time)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Car Price Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { font-weight: bold; color: #007bff; }
            .path { font-family: monospace; background: #e9ecef; padding: 2px 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 Car Price Prediction API</h1>
            <p>Welcome to the Car Price Prediction API! This service provides machine learning-based predictions for car prices.</p>
            
            <h2>Available Endpoints</h2>
            
            <div class="endpoint">
                <div class="method">GET</div>
                <div class="path">/</div>
                <p>This documentation page</p>
            </div>
            
            <div class="endpoint">
                <div class="method">GET</div>
                <div class="path">/health</div>
                <p>Health check endpoint</p>
            </div>
            
            <div class="endpoint">
                <div class="method">POST</div>
                <div class="path">/predict</div>
                <p>Predict price for a single car</p>
            </div>
            
            <div class="endpoint">
                <div class="method">POST</div>
                <div class="path">/predict/batch</div>
                <p>Upload CSV file for batch predictions</p>
            </div>
            
            <div class="endpoint">
                <div class="method">GET</div>
                <div class="path">/docs</div>
                <p>Interactive API documentation (Swagger UI)</p>
            </div>
            
            <h2>Usage Examples</h2>
            
            <h3>Single Prediction</h3>
            <pre>
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "doorNumber": 5,
       "vehicle_age_years": 3,
       "km_per_year": 15000,
       "sale_year": 2024,
       "sale_month": 6,
       "years_since_intro_at_sale": 5,
       "kilometers": 45000,
       "cubeCapacity": 1200,
       "powerKW": 80,
       "cylinder": 4,
       "is_neutral_color": 1,
       "aestheticGrade_ordinal": 3,
       "mechanicalGrade_ordinal": 3,
       "type__Hatchback": 1,
       "type__Estate": 0,
       "type__Sedan": 0,
       "fuel__Diesel": 0,
       "fuel__Petrol": 1,
       "transmission__Manual": 1,
       "make_model__te": 5000
     }'
            </pre>
            
            <h3>Batch Prediction</h3>
            <pre>
curl -X POST "http://localhost:8000/predict/batch" \\
     -F "file=@cars.csv"
            </pre>
            
            <p><strong>Note:</strong> For batch predictions, upload a CSV file with the same column structure as the training data.</p>
        </div>
    </body>
    </html>
    """
    return html_content


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "preprocessor_loaded": model_service.preprocessor is not None,
        "transformers_loaded": model_service.transformers is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Predict price for a single car."""
    try:
        # Convert request to dict
        data = request.dict()
        
        # Make prediction
        prediction = model_service.predict_single(data)
        
        return PredictionResponse(
            prediction=prediction,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """Predict prices for a batch of cars from CSV file."""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))
        
        # Make predictions
        predictions = model_service.predict_batch(df)
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if model_service.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    info = {
        "model_type": type(model_service.model).__name__,
        "feature_count": len(model_service.feature_names) if model_service.feature_names else "Unknown",
        "feature_names": model_service.feature_names,
        "model_path": model_service.model_path,
        "preprocessor_path": model_service.preprocessor_path,
        "transformers_path": model_service.transformers_path,
        "target_transformed": model_service.transformers.get('target_log_transformed', False) if model_service.transformers else False,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add model-specific information
    if hasattr(model_service.model, 'n_estimators'):
        info["n_estimators"] = int(model_service.model.n_estimators)
    if hasattr(model_service.model, 'learning_rate'):
        info["learning_rate"] = float(model_service.model.learning_rate)
    if hasattr(model_service.model, 'feature_importances_'):
        # Get top 10 most important features
        importances = model_service.model.feature_importances_
        feature_importance = dict(zip(model_service.feature_names, importances))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        info["top_features"] = [(k, float(v)) for k, v in top_features]
    
    return info


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from fastapi.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)