"""
Unit tests for the car price prediction system.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.price_model.transformers import DateAndMileage, EngineFeatures, RareCategoryGrouper
from src.price_model.num_sanitize import NumericSanitizer, ColCfg
from src.price_model.evaluation import ModelEvaluator
from src.price_model.model_comparison import ModelComparison


class TestDateAndMileage:
    """Test the DateAndMileage transformer."""
    
    def test_basic_functionality(self):
        """Test basic date and mileage feature creation."""
        transformer = DateAndMileage()
        
        # Create test data
        data = pd.DataFrame({
            'saleDate': ['2022-01-01', '2022-01-01', '2022-01-01'],
            'registrationDate': ['2020-01-01', '2018-01-01', '2015-01-01'],
            'kilometers': [10000, 50000, 100000]
        })
        
        result = transformer.fit_transform(data)
        
        # Check that new features are created
        assert 'age_years' in result.columns
        assert 'km_per_year' in result.columns
        assert 'sale_year' in result.columns
        assert 'sale_month' in result.columns
        
        # Check age calculation
        expected_ages = [2.0, 4.0, 7.0]  # Approximate
        np.testing.assert_array_almost_equal(result['age_years'].values, expected_ages, decimal=1)
    
    def test_handles_invalid_dates(self):
        """Test handling of invalid dates."""
        transformer = DateAndMileage()
        
        data = pd.DataFrame({
            'saleDate': ['2022-01-01', 'invalid', '2022-01-01'],
            'registrationDate': ['2020-01-01', '2018-01-01', 'invalid'],
            'kilometers': [10000, 50000, 100000]
        })
        
        result = transformer.fit_transform(data)
        
        # Should handle invalid dates gracefully
        assert not result['age_years'].isna().all()
        assert not result['km_per_year'].isna().all()


class TestEngineFeatures:
    """Test the EngineFeatures transformer."""
    
    def test_basic_functionality(self):
        """Test basic engine feature creation."""
        transformer = EngineFeatures()
        
        data = pd.DataFrame({
            'cubeCapacity': [1600, 2000, 3000],
            'powerKW': [100, 150, 200],
            'powerHP': [136, 204, 272]
        })
        
        result = transformer.fit_transform(data)
        
        # Check that powerHP is dropped
        assert 'powerHP' not in result.columns
        
        # Check that power_per_liter is created
        assert 'power_per_liter' in result.columns
        
        # Check power_per_liter calculation
        expected_power_per_liter = [100 / 1.6, 150 / 2.0, 200 / 3.0]
        np.testing.assert_array_almost_equal(result['power_per_liter'].values, expected_power_per_liter)
    
    def test_handles_invalid_capacity(self):
        """Test handling of invalid engine capacity values."""
        transformer = EngineFeatures()
        
        data = pd.DataFrame({
            'cubeCapacity': [1600, -100, 0, 10000],  # Invalid values
            'powerKW': [100, 150, 200, 300]
        })
        
        result = transformer.fit_transform(data)
        
        # Invalid values should be converted to NaN
        assert result['cubeCapacity'].iloc[1] is np.nan
        assert result['cubeCapacity'].iloc[2] is np.nan
        assert result['cubeCapacity'].iloc[3] is np.nan


class TestRareCategoryGrouper:
    """Test the RareCategoryGrouper transformer."""
    
    def test_basic_functionality(self):
        """Test basic rare category grouping."""
        transformer = RareCategoryGrouper(['make'], min_count=2)
        
        # Create test data with rare categories
        data = pd.DataFrame({
            'make': ['Toyota', 'Toyota', 'Toyota', 'BMW', 'BMW', 'Ferrari', 'Lamborghini']
        })
        
        result = transformer.fit_transform(data)
        
        # Rare categories should be grouped as 'Other'
        unique_makes = result['make'].unique()
        assert 'Other' in unique_makes
        assert 'Ferrari' not in unique_makes
        assert 'Lamborghini' not in unique_makes
    
    def test_fit_transform_separately(self):
        """Test fit and transform called separately."""
        transformer = RareCategoryGrouper(['make'], min_count=2)
        
        train_data = pd.DataFrame({
            'make': ['Toyota', 'Toyota', 'Toyota', 'BMW', 'BMW', 'Ferrari']
        })
        
        test_data = pd.DataFrame({
            'make': ['Toyota', 'BMW', 'Ferrari', 'Lamborghini']  # New rare category
        })
        
        # Fit on training data
        transformer.fit(train_data)
        
        # Transform test data
        result = transformer.transform(test_data)
        
        # New rare category should be grouped as 'Other'
        assert result['make'].iloc[3] == 'Other'


class TestNumericSanitizer:
    """Test the NumericSanitizer transformer."""
    
    def test_basic_functionality(self):
        """Test basic numeric sanitization."""
        config = {
            'kilometers': ColCfg(
                cap_method='quantile',
                q=0.9,
                skew='log1p',
                impute='median',
                add_indicator=True
            )
        }
        
        transformer = NumericSanitizer(config)
        
        # Create test data with outliers
        data = pd.DataFrame({
            'kilometers': [10000, 50000, 100000, 500000, 1000000]  # Last value is outlier
        })
        
        result = transformer.fit_transform(data)
        
        # Check that new features are created
        assert 'kilometers_is_missing' in result.columns
        assert 'log1p_kilometers' in result.columns
        
        # Check that outlier is capped
        assert result['log1p_kilometers'].max() < np.log1p(1000000)
    
    def test_domain_bounds(self):
        """Test domain bounds enforcement."""
        config = {
            'kilometers': ColCfg(
                domain_max=500000,
                impute='median'
            )
        }
        
        transformer = NumericSanitizer(config)
        
        data = pd.DataFrame({
            'kilometers': [10000, 50000, 100000, 600000]  # Last value exceeds domain
        })
        
        result = transformer.fit_transform(data)
        
        # Value exceeding domain should be NaN
        assert result['kilometers'].iloc[3] is np.nan


class TestModelEvaluator:
    """Test the ModelEvaluator class."""
    
    def test_calculate_metrics(self):
        """Test metric calculation."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        # Check that all expected metrics are present
        expected_metrics = ['rmse', 'mae', 'mdae', 'mape', 'r2', 'mape_robust', 'smape', 'wmape']
        for metric in expected_metrics:
            assert metric in metrics
            assert not np.isnan(metrics[metric])
    
    def test_handles_nan_values(self):
        """Test handling of NaN values in predictions."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, np.nan, 310, 390, 510])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        # Should handle NaN values gracefully
        assert not np.isnan(metrics['rmse'])
        assert not np.isnan(metrics['mae'])


class TestModelComparison:
    """Test the model comparison framework."""
    
    def test_model_comparison_builds(self):
        """Test that the model comparison framework builds without errors."""
        model_comparison = ModelComparison()
        base_models = model_comparison.get_base_models()
        
        # Check that we have expected models
        expected_models = ['LightGBM', 'XGBoost', 'RandomForest', 'GradientBoosting', 'Ridge']
        for model in expected_models:
            assert model in base_models
    
    def test_pipeline_creation(self):
        """Test that pipelines can be created for each model."""
        model_comparison = ModelComparison()
        base_models = model_comparison.get_base_models()
        
        # Test that we can build pipelines for each model
        pipelines = model_comparison.build_model_pipelines(base_models)
        
        # Check that pipelines were created
        assert len(pipelines) > 0
        assert all(hasattr(pipeline, 'fit') for pipeline in pipelines.values())
        assert all(hasattr(pipeline, 'predict') for pipeline in pipelines.values())


if __name__ == "__main__":
    pytest.main([__file__])
