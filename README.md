# Car Price Prediction System

A machine learning pipeline for predicting car prices using time series data with proper cross-validation and no data leakage.

## Features

- **Time Series Cross-Validation**: Proper handling of temporal data without leakage
- **Feature Engineering**: Domain-specific feature creation and encoding
- **Multicollinearity Pruning**: Automated feature selection using correlation and VIF
- **Model Comparison**: Multiple algorithms with ensemble methods
- **Hyperparameter Optimization**: Bayesian optimization with Optuna
- **Experiment Tracking**: MLFlow integration for model versioning
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
├── multicollinearity_pruner.py  # Feature selection and VIF analysis
├── modeling_pipeline.py         # Main ML pipeline with time series CV
├── experiment_tracker.py        # MLFlow integration
├── model_service.py            # FastAPI service
├── model_comparison.py         # Model evaluation framework
└── hyperparameter_optimization.py # Bayesian optimization
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

- **API Documentation**: http://localhost:8000/docs
- **MLFlow UI**: http://localhost:5000
- **Codebase Guide**: [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md)

## Requirements

- Python 3.8+
- Docker (for deployment)
- 8GB+ RAM recommended

## License

MIT License
