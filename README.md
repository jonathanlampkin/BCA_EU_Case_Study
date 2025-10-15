# Car Price Prediction System

> Start here: See the full guide and instructions in [docs/Project_Documentation.md](docs/Project_Documentation.md).

A machine learning pipeline for predicting car prices using time series data with proper cross-validation and no data leakage.

## Features

- **Time Series Cross-Validation**: Proper handling of temporal data without leakage
- **Feature Engineering**: Domain-specific feature creation and encoding
- **Multicollinearity Pruning**: Automated feature selection using correlation and VIF
- **Model Comparison**: Multiple algorithms with ensemble methods
- **Hyperparameter Optimization**: Bayesian optimization with Optuna
- **Experiment Tracking**: MLflow integration; runs logged under project root `mlruns/`
- **REST API**: FastAPI service for real-time predictions
- **Production Ready**: Docker deployment and monitoring

## Quick Start

```bash
# Setup environment
make setup

# Run complete pipeline
make full

# Start API service
make serve
```

Notes:
- Artifacts under `artifacts/` are regenerated; previous run outputs are cleaned at pipeline start.
- MLflow logs are persisted locally to the repository’s root `mlruns/` directory.

## API Usage

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
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
  "type_Hatchback": 1,
  "type_Estate": 0,
  "type_Sedan": 0,
  "fuel_Diesel": 0,
  "fuel_Petrol": 1,
  "transmission_Manual": 1,
  "make_model_te": 5000
}'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
-F "file=@cars.csv"
```

## Model Performance

- **Best Model**: XGBoost_Conservative (Optimized)
- **Test RMSE**: 2,837.54
- **Test R²**: 0.795 (79.5% variance explained)
- **Features**: 11 features after multicollinearity pruning

## Architecture

```
src/price_model/
├── preprocessor.py              # Data preprocessing and feature engineering
├── advanced_feature_selector.py # Correlation/VIF-driven selection and summaries
├── feature_analyzer.py          # SHAP/importance plots and reporting
├── transformers.py              # Utilities for transforms
├── num_sanitize.py              # Numeric sanitization helpers
├── modeling_pipeline.py         # End-to-end model pipeline (CV, optimize, train, save)
├── experiment_tracker.py        # MLflow integration (logs → root mlruns/)
└── model_service.py             # FastAPI service for predictions
```

## Development

### Setup
```bash
make setup
```

### Run Tests
```bash
make test
```

### Code Quality
```bash
make lint
make format
```

### Pipeline Commands
```bash
make validate    # Data validation
make preprocess  # Data preprocessing
make model       # Model training
make train       # Final model training
make serve       # Start API service
```

## Deployment

### Docker
```bash
make deploy
```

### Azure
- Optimized Dockerfile included
- Kubernetes YAML configurations
- Environment variable setup

## Documentation

- **Project Guide (Start Here)**: [docs/Project_Documentation.md](docs/Project_Documentation.md)
- **API Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000

## Requirements

- Python 3.8+
- Docker (for deployment)
- 8GB+ RAM recommended

## License

MIT License
