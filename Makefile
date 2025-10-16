# Configuration
PYTHON := python3
DATA_PATH := data/EUDS_CaseStudy_Pricing.csv
ARTIFACTS_DIR := artifacts
SRC_DIR := src
VENV_DIR := venv
KERNEL_NAME := bca-eu

# Cross-platform venv python/pip
ifeq ($(OS),Windows_NT)
  VENV_PY := $(VENV_DIR)/Scripts/python.exe
  VENV_PIP := $(VENV_DIR)/Scripts/pip.exe
  ACTIVATE_CMD := cmd.exe /k "$(VENV_DIR)\\Scripts\\activate"
else
  VENV_PY := $(VENV_DIR)/bin/python
  VENV_PIP := $(VENV_DIR)/bin/pip
  ACTIVATE_CMD := /usr/bin/bash -i -c "source $(VENV_DIR)/bin/activate && exec /usr/bin/bash -i"
endif

# Pipeline parameters
CV_FOLDS := 3
N_TRIALS := 30
VIF_THRESHOLD := 10.0
CORRELATION_THRESHOLD := 0.80

.PHONY: help preprocess model serve full setup


help:
	@echo "Car Price Prediction Pipeline"
	@echo "============================="
	@echo
	@echo "Available commands:"
	@echo "  setup        Full setup: venv + deps + kernel + open activated shell"
	@echo "  preprocess   Clean and preprocess the raw data"
	@echo "  model        Train and evaluate the models"
	@echo "  serve        Start the API for predictions"
	@echo "  full         Run preprocess, model, then serve"
	@echo
	@echo "Notes:"
	@echo "- Data input: $(DATA_PATH)"
	@echo "- Clean data output: data/clean_data.csv"
	@echo "- Artifacts (models, reports, outputs): artifacts/"


setup: ## Full setup: create venv, install deps, register Jupyter kernel, open activated shell
	@echo "Setting up development environment..."
	@echo "- Creating virtual environment if it doesn't exist"
	@if [ ! -d "$(VENV_DIR)" ]; then $(PYTHON) -m venv $(VENV_DIR); fi
	@echo "- Installing requirements into the virtual environment"
	@"$(VENV_PY)" -m pip install --upgrade pip
	@"$(VENV_PY)" -m pip install -r requirements.txt -r requirements-dev.txt
	@echo "Setup complete."
	@echo
	@echo "Registering/refreshing Jupyter kernel for this environment..."
	@"$(VENV_PY)" -m ipykernel install --user --name "$(KERNEL_NAME)" --display-name "Python ($(KERNEL_NAME))" >/dev/null 2>&1 || true
	@echo "Kernel 'Python ($(KERNEL_NAME))' is available in Jupyter."
	@echo
	@echo "Opening an activated shell in this environment..."
	@$(ACTIVATE_CMD)


 


register-kernel: ## Register this venv as a Jupyter kernel
	@echo "Registering Jupyter kernel for this virtual environment..."
	@"$(VENV_PY)" -m ipykernel install --user --name "$(KERNEL_NAME)" --display-name "Python ($(KERNEL_NAME))"
	@echo "Kernel 'Python ($(KERNEL_NAME))' registered. Select it in Jupyter for notebooks."


notebook: ## Launch Jupyter Lab using the venv interpreter
	@echo "Launching Jupyter Lab using the virtual environment..."
	@"$(VENV_PY)" -m jupyter lab


preprocess: ## Clean and preprocess the data
	@echo "Preprocessing data"
	@mkdir -p data
	@$(VENV_PY) -m src.price_model.preprocessor \
		--input_path $(DATA_PATH) \
		--output_path data/clean_data.csv
	@echo "Data preprocessing complete"
	@echo "Verifying data structure..."
	@$(VENV_PY) -c "import pandas as pd; df = pd.read_csv('data/clean_data.csv'); \
		print('Data shape:', df.shape); \
		print('Data types:'); \
		print(df.dtypes.value_counts()); \
		null_count = int(df.isnull().sum().sum()); \
		print('Null count:', null_count)"


model: ## Run modeling pipeline (requires preprocessed data)
	@echo "Running comprehensive modeling pipeline..."
	@test -f data/clean_data.csv || (echo "Missing data/clean_data.csv. Run 'make preprocess' first." && exit 1)
	@$(VENV_PY) src/price_model/modeling_pipeline.py \
			--data data/clean_data.csv \
			--cv-folds $(CV_FOLDS) \
			--correlation-threshold $(CORRELATION_THRESHOLD) \
			--vif-threshold $(VIF_THRESHOLD) \
			--n-trials $(N_TRIALS)
	@echo "Modeling pipeline complete!"

# API service
serve: ## Start FastAPI service (requires trained artifacts)
	@echo "Starting FastAPI service..."
	@echo "API will be available at: http://localhost:8000"
	@echo "API documentation at: http://localhost:8000/docs"
	@echo "Upload CSV for batch predictions at: http://localhost:8000/predict/batch"
	@test -f artifacts/final_model.joblib || (echo "Missing artifacts/final_model.joblib. Run 'make model' first." && exit 1)
	@test -f artifacts/preprocessor.joblib || (echo "Missing artifacts/preprocessor.joblib. Run 'make model' first." && exit 1)
	@test -f artifacts/transformers.joblib || (echo "Missing artifacts/transformers.joblib. Run 'make model' first." && exit 1)
	@$(VENV_PY) src/price_model/model_service.py

# Full pipeline

full: ## Run preprocess -> model -> serve
	$(MAKE) preprocess
	$(MAKE) model
	@echo "Generating seaborn comparison charts..."
	@$(VENV_PY) scripts/plot_model_comparison.py
	$(MAKE) serve
	@echo "Done. API running at http://localhost:8000"