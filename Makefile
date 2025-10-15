# Car Price Prediction Pipeline Makefile
# Simplified with only required commands

# Configuration
PYTHON := python3
PIP := pip3
DATA_PATH := data/EUDS_CaseStudy_Pricing.csv
ARTIFACTS_DIR := artifacts
SRC_DIR := src
VENV_DIR := venv

# Pipeline parameters
CV_FOLDS := 3
N_TRIALS := 30
VIF_THRESHOLD := 10.0
CORRELATION_THRESHOLD := 0.80

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

.PHONY: help preprocess model serve full

# Default target
help: ## Show this help message
	@echo "$(BLUE)Car Price Prediction Pipeline$(NC)"
	@echo "$(BLUE)============================$(NC)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  make full           # Run complete pipeline from start to finish"
	@echo ""
	@echo "$(YELLOW)Step-by-Step:$(NC)"
	@echo "  1. make preprocess  # Clean and preprocess data with OHE for color"
	@echo "  2. make model       # Run comprehensive modeling pipeline"
	@echo "  3. make serve       # Start FastAPI service for real-time predictions"
	@echo ""
	@echo "$(YELLOW)Pipeline Flow:$(NC)"
	@echo "  make full           # Run all 3 commands in sequence"

# Environment setup
setup: ## Set up the development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "$(YELLOW)Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@echo "$(YELLOW)Activating virtual environment and installing dependencies...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PIP) install --upgrade pip && \
		$(PIP) install -r requirements.txt && \
		$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)Environment setup complete!$(NC)"
	@echo "$(YELLOW)To activate: source $(VENV_DIR)/bin/activate$(NC)"

# Data preprocessing
preprocess: ## Clean and preprocess the data with OHE for color
	@echo "$(BLUE)Preprocessing data with OHE for color...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) src/price_model/preprocessor.py \
			--input_path $(DATA_PATH) \
			--output_path $(ARTIFACTS_DIR)/clean_data.csv
	@echo "$(GREEN)Data preprocessing complete!$(NC)"
	@echo "$(YELLOW)Verifying data structure...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) -c "import pandas as pd; df = pd.read_csv('$(ARTIFACTS_DIR)/clean_data.csv'); \
		print('Data shape:', df.shape); \
		print('Columns:', list(df.columns)); \
		print('Data types:'); \
		print(df.dtypes.value_counts()); \
		print('All numeric columns (except saleDate):', [col for col in df.columns if col != 'saleDate' and pd.api.types.is_numeric_dtype(df[col])])"

# Modeling pipeline
model: preprocess ## Run comprehensive modeling pipeline with all improvements
	@echo "$(BLUE)Running comprehensive modeling pipeline...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) src/price_model/modeling_pipeline.py \
			--data $(ARTIFACTS_DIR)/clean_data.csv \
			--cv-folds $(CV_FOLDS) \
			--correlation-threshold $(CORRELATION_THRESHOLD) \
			--vif-threshold $(VIF_THRESHOLD) \
			--n-trials $(N_TRIALS)
	@echo "$(GREEN)Modeling pipeline complete!$(NC)"

# API service
serve: model ## Start FastAPI service for real-time predictions
	@echo "$(BLUE)Starting FastAPI service...$(NC)"
	@echo "$(YELLOW)API will be available at: http://localhost:8000$(NC)"
	@echo "$(YELLOW)API documentation at: http://localhost:8000/docs$(NC)"
	@echo "$(YELLOW)Upload CSV for batch predictions at: http://localhost:8000/predict/batch$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) src/price_model/model_service.py

# Full pipeline
full: serve ## Run complete pipeline from preprocessing to serving
	@echo "$(GREEN)🎉 Complete pipeline finished!$(NC)"
	@echo "$(YELLOW)API is running at: http://localhost:8000$(NC)"
	@echo "$(YELLOW)API docs at: http://localhost:8000/docs$(NC)"