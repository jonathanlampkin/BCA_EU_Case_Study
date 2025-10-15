"""
Integration tests for the car price prediction system.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from unittest.mock import patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.price_model.pipeline_orchestrator import PipelineOrchestrator
from src.price_model.model_comparison import ModelComparison
from src.price_model.hyperparameter_optimization import BayesianOptimizer
from src.price_model.evaluation import ModelEvaluator


class TestPipelineOrchestrator:
    """Test the complete pipeline orchestrator."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'vehicleID': [f'V_{i}' for i in range(n_samples)],
            'registrationDate': pd.date_range('2015-01-01', periods=n_samples, freq='D'),
            'saleDate': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            'kilometers': np.random.lognormal(10, 0.5, n_samples).astype(int),
            'colour': np.random.choice(['Red', 'Blue', 'Black', 'White', 'Silver'], n_samples),
            'aestheticGrade': np.random.choice(['Bad', 'Medium', 'Good', 'Very Good'], n_samples),
            'mechanicalGrade': np.random.choice(['Bad', 'Medium', 'Good', 'Very Good'], n_samples),
            'make': np.random.choice(['Toyota', 'BMW', 'Mercedes', 'Audi', 'Volkswagen'], n_samples),
            'model': np.random.choice(['Camry', 'X3', 'C-Class', 'A4', 'Golf'], n_samples),
            'doorNumber': np.random.choice([3, 4, 5], n_samples),
            'type': np.random.choice(['Sedan', 'SUV', 'Hatchback', 'Estate'], n_samples),
            'fuel': np.random.choice(['Petrol', 'Diesel', 'Hybrid'], n_samples),
            'transmission': np.random.choice(['Manual', 'Automatic'], n_samples),
            'yearIntroduced': np.random.randint(2010, 2020, n_samples),
            'cylinder': np.random.choice([4, 6, 8], n_samples),
            'cubeCapacity': np.random.choice([1600, 2000, 3000, 4000], n_samples),
            'powerKW': np.random.normal(120, 30, n_samples).astype(int),
            'powerHP': np.random.normal(160, 40, n_samples).astype(int),
            'targetPrice': np.random.lognormal(9, 0.5, n_samples).astype(int)
        })
        
        return data
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_data_validation(self, sample_data, temp_dir):
        """Test data validation functionality."""
        # Save sample data
        data_path = os.path.join(temp_dir, 'test_data.csv')
        sample_data.to_csv(data_path, index=False)
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(data_path, temp_dir)
        
        # Run validation
        validation_results = orchestrator.load_and_validate_data()
        
        # Check validation results
        assert 'data_shape' in validation_results
        assert 'missing_values' in validation_results
        assert 'data_quality_score' in validation_results
        assert validation_results['data_quality_score'] > 0
        
        # Check that validation report was saved
        assert os.path.exists(os.path.join(temp_dir, 'data_validation_report.json'))
    
    def test_data_preprocessing(self, sample_data, temp_dir):
        """Test data preprocessing functionality."""
        # Save sample data
        data_path = os.path.join(temp_dir, 'test_data.csv')
        sample_data.to_csv(data_path, index=False)
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(data_path, temp_dir)
        
        # Run preprocessing
        orchestrator.load_and_validate_data()
        clean_data = orchestrator.preprocess_data()
        
        # Check that clean data was saved
        assert os.path.exists(os.path.join(temp_dir, 'clean_data.csv'))
        
        # Check that clean data has expected shape
        assert clean_data.shape[0] <= sample_data.shape[0]  # Some rows might be dropped
        
        # Check that date columns are properly converted
        assert pd.api.types.is_datetime64_any_dtype(clean_data['registrationDate'])
        assert pd.api.types.is_datetime64_any_dtype(clean_data['saleDate'])
    
    def test_data_splitting(self, sample_data, temp_dir):
        """Test data splitting functionality."""
        # Save sample data
        data_path = os.path.join(temp_dir, 'test_data.csv')
        sample_data.to_csv(data_path, index=False)
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(data_path, temp_dir)
        
        # Run preprocessing and splitting
        orchestrator.load_and_validate_data()
        orchestrator.preprocess_data()
        split_info = orchestrator.split_data(test_months=3)
        
        # Check split info
        assert 'train_shape' in split_info
        assert 'test_shape' in split_info
        assert 'cutoff_date' in split_info
        
        # Check that train and test data are available
        assert orchestrator.train_data is not None
        assert orchestrator.test_data is not None
        
        # Check that split makes sense
        assert len(orchestrator.train_data) > len(orchestrator.test_data)
        assert len(orchestrator.train_data) + len(orchestrator.test_data) == len(orchestrator.clean_data)
    
    def test_model_comparison(self, sample_data, temp_dir):
        """Test model comparison functionality."""
        # Save sample data
        data_path = os.path.join(temp_dir, 'test_data.csv')
        sample_data.to_csv(data_path, index=False)
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(data_path, temp_dir)
        
        # Run preprocessing and splitting
        orchestrator.load_and_validate_data()
        orchestrator.preprocess_data()
        orchestrator.split_data(test_months=3)
        
        # Run model comparison (with fewer CV folds for speed)
        comparison_results = orchestrator.compare_models(cv_folds=3)
        
        # Check results
        assert len(comparison_results) > 0
        assert 'model' in comparison_results.columns
        assert 'rmse' in comparison_results.columns
        assert 'mae' in comparison_results.columns
        assert 'r2' in comparison_results.columns
        
        # Check that results are sorted by RMSE
        assert comparison_results['rmse'].is_monotonic_increasing
        
        # Check that best model is identified
        assert orchestrator.best_model is not None
        assert orchestrator.best_model in comparison_results['model'].values
    
    def test_timeseries_feasibility(self, sample_data, temp_dir):
        """Test time series feasibility analysis."""
        # Save sample data
        data_path = os.path.join(temp_dir, 'test_data.csv')
        sample_data.to_csv(data_path, index=False)
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(data_path, temp_dir)
        
        # Run preprocessing
        orchestrator.load_and_validate_data()
        orchestrator.preprocess_data()
        
        # Run time series feasibility analysis
        feasibility_results = orchestrator.evaluate_timeseries_feasibility()
        
        # Check results
        assert 'recommendation' in feasibility_results
        assert isinstance(feasibility_results['recommendation'], str)
        
        # Check that analysis was saved
        assert os.path.exists(os.path.join(temp_dir, 'timeseries_feasibility_analysis.json'))


class TestModelComparison:
    """Test the model comparison framework."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 500  # Smaller for faster testing
        
        data = pd.DataFrame({
            'registrationDate': pd.date_range('2015-01-01', periods=n_samples, freq='D'),
            'saleDate': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            'kilometers': np.random.lognormal(10, 0.5, n_samples).astype(int),
            'cubeCapacity': np.random.choice([1600, 2000, 3000], n_samples),
            'powerKW': np.random.normal(120, 30, n_samples).astype(int),
            'make': np.random.choice(['Toyota', 'BMW', 'Mercedes'], n_samples),
            'model': np.random.choice(['Camry', 'X3', 'C-Class'], n_samples),
            'colour': np.random.choice(['Red', 'Blue', 'Black'], n_samples),
            'fuel': np.random.choice(['Petrol', 'Diesel'], n_samples),
            'transmission': np.random.choice(['Manual', 'Automatic'], n_samples),
            'type': np.random.choice(['Sedan', 'SUV'], n_samples),
            'aestheticGrade': np.random.choice(['Good', 'Very Good'], n_samples),
            'mechanicalGrade': np.random.choice(['Good', 'Very Good'], n_samples),
            'doorNumber': np.random.choice([4, 5], n_samples),
            'targetPrice': np.random.lognormal(9, 0.5, n_samples).astype(int)
        })
        
        return data
    
    def test_model_comparison_basic(self, sample_data):
        """Test basic model comparison functionality."""
        # Prepare data
        X = sample_data.drop(columns=['targetPrice'])
        y = sample_data['targetPrice']
        
        # Initialize model comparison
        model_comparison = ModelComparison(cv_folds=3)  # Fewer folds for speed
        
        # Run comparison
        results = model_comparison.compare_models(X, y)
        
        # Check results
        assert len(results) > 0
        assert 'model' in results.columns
        assert 'rmse' in results.columns
        
        # Check that all models were evaluated
        expected_models = ['LightGBM', 'XGBoost', 'RandomForest', 'GradientBoosting', 'Ridge']
        for model in expected_models:
            assert model in results['model'].values
    
    def test_get_best_model(self, sample_data):
        """Test getting the best model."""
        # Prepare data
        X = sample_data.drop(columns=['targetPrice'])
        y = sample_data['targetPrice']
        
        # Initialize model comparison
        model_comparison = ModelComparison(cv_folds=3)
        
        # Run comparison
        model_comparison.compare_models(X, y)
        
        # Get best model
        best_model, best_rmse = model_comparison.get_best_model()
        
        # Check results
        assert best_model is not None
        assert best_rmse > 0
        assert isinstance(best_model, str)
        assert isinstance(best_rmse, float)


class TestBayesianOptimization:
    """Test the Bayesian optimization framework."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 200  # Small for fast testing
        
        data = pd.DataFrame({
            'registrationDate': pd.date_range('2015-01-01', periods=n_samples, freq='D'),
            'saleDate': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            'kilometers': np.random.lognormal(10, 0.5, n_samples).astype(int),
            'cubeCapacity': np.random.choice([1600, 2000, 3000], n_samples),
            'powerKW': np.random.normal(120, 30, n_samples).astype(int),
            'make': np.random.choice(['Toyota', 'BMW'], n_samples),
            'model': np.random.choice(['Camry', 'X3'], n_samples),
            'colour': np.random.choice(['Red', 'Blue'], n_samples),
            'fuel': np.random.choice(['Petrol', 'Diesel'], n_samples),
            'transmission': np.random.choice(['Manual', 'Automatic'], n_samples),
            'type': np.random.choice(['Sedan', 'SUV'], n_samples),
            'aestheticGrade': np.random.choice(['Good', 'Very Good'], n_samples),
            'mechanicalGrade': np.random.choice(['Good', 'Very Good'], n_samples),
            'doorNumber': np.random.choice([4, 5], n_samples),
            'targetPrice': np.random.lognormal(9, 0.5, n_samples).astype(int)
        })
        
        return data
    
    def test_lightgbm_optimization(self, sample_data):
        """Test LightGBM hyperparameter optimization."""
        # Prepare data
        X = sample_data.drop(columns=['targetPrice'])
        y = sample_data['targetPrice']
        
        # Initialize optimizer
        optimizer = BayesianOptimizer(n_trials=5)  # Few trials for speed
        
        # Run optimization
        results = optimizer.optimize_lightgbm(X, y)
        
        # Check results
        assert 'best_params' in results
        assert 'best_score' in results
        assert 'n_trials' in results
        
        # Check that optimization improved
        assert results['best_score'] > 0
        assert results['n_trials'] == 5
    
    def test_ridge_optimization(self, sample_data):
        """Test Ridge hyperparameter optimization."""
        # Prepare data
        X = sample_data.drop(columns=['targetPrice'])
        y = sample_data['targetPrice']
        
        # Initialize optimizer
        optimizer = BayesianOptimizer(n_trials=5)
        
        # Run optimization
        results = optimizer.optimize_ridge(X, y)
        
        # Check results
        assert 'best_params' in results
        assert 'best_score' in results
        assert 'alpha' in results['best_params']


class TestModelEvaluator:
    """Test the model evaluator."""
    
    def test_cross_validation(self):
        """Test cross-validation functionality."""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples)
        })
        
        y = pd.Series(np.random.normal(100, 20, n_samples))
        
        # Create a simple pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LinearRegression
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['feature1', 'feature2']),
                ('cat', OneHotEncoder(), ['feature3'])
            ]
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])
        
        # Initialize evaluator
        evaluator = ModelEvaluator(cv_folds=3)
        
        # Run cross-validation
        results = evaluator.cross_validate(pipeline, X, y, 'test_model')
        
        # Check results
        assert 'mean' in results
        assert 'std' in results
        assert 'fold_results' in results
        
        # Check that metrics are calculated
        assert 'rmse' in results['mean']
        assert 'mae' in results['mean']
        assert 'r2' in results['mean']
        
        # Check that results are reasonable
        assert results['mean']['rmse'] > 0
        assert results['mean']['r2'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
