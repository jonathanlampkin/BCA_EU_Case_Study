"""
Experiment tracking and model versioning with MLFlow.
"""
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime


class ExperimentTracker:
    """MLFlow experiment tracking for car price prediction models."""
    
    def __init__(self, experiment_name: str = "car_price_prediction"):
        self.experiment_name = experiment_name
        self._setup_experiment()
    
    def _setup_experiment(self):
        """Initialize MLFlow experiment and ensure local tracking under project root mlruns."""
        # Resolve project root from this file: src/price_model/ -> src -> project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        local_mlruns_path = os.path.join(project_root, 'mlruns')
        # Use file URI to avoid ambiguity about working directory
        mlflow.set_tracking_uri(f"file://{local_mlruns_path}")
        mlflow.set_experiment(self.experiment_name)
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLFlow run."""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return mlflow.start_run(run_name=run_name)
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log model parameters."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log model metrics."""
        mlflow.log_metrics(metrics)
    
    def log_model(self, model, model_name: str, signature=None):
        """Log trained model."""
        if hasattr(model, 'predict'):
            if 'lightgbm' in str(type(model)).lower():
                mlflow.lightgbm.log_model(model, model_name, signature=signature)
            elif 'xgboost' in str(type(model)).lower():
                mlflow.xgboost.log_model(model, model_name, signature=signature)
            else:
                mlflow.sklearn.log_model(model, model_name, signature=signature)
    
    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
        """Log file artifact."""
        mlflow.log_artifact(file_path, artifact_path)
    
    def log_data_info(self, X_shape: tuple, y_shape: tuple, feature_names: list):
        """Log dataset information."""
        mlflow.log_params({
            "n_samples": X_shape[0],
            "n_features": X_shape[1],
            "target_shape": y_shape[0],
            "feature_count": len(feature_names)
        })
        
        # Log feature names as artifact
        with open("feature_names.json", "w") as f:
            json.dump(feature_names, f, indent=2)
        mlflow.log_artifact("feature_names.json")
        os.remove("feature_names.json")
    
    def log_preprocessing_info(self, preprocessing_steps: Dict[str, Any]):
        """Log preprocessing configuration."""
        mlflow.log_params(preprocessing_steps)
    
    def end_run(self):
        """End current run."""
        mlflow.end_run()
