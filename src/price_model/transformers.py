"""
Minimal transformers for the preprocessor.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional

class DateAndMileage(BaseEstimator, TransformerMixin):
    """Extract date and mileage features."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Convert saleDate to datetime
        if 'saleDate' in X.columns:
            X['saleDate'] = pd.to_datetime(X['saleDate'], errors='coerce')
            X['sale_year'] = X['saleDate'].dt.year
            X['sale_month'] = X['saleDate'].dt.month
        
        # Calculate vehicle age
        if 'registrationDate' in X.columns and 'saleDate' in X.columns:
            X['registrationDate'] = pd.to_datetime(X['registrationDate'], errors='coerce')
            X['vehicle_age_years'] = (X['saleDate'] - X['registrationDate']).dt.days / 365.25
        
        # Calculate km per year
        if 'kilometers' in X.columns and 'vehicle_age_years' in X.columns:
            X['km_per_year'] = X['kilometers'] / X['vehicle_age_years'].replace(0, np.nan)
        
        # Calculate years since introduction
        if 'yearIntroduced' in X.columns and 'sale_year' in X.columns:
            X['years_since_intro_at_sale'] = X['sale_year'] - X['yearIntroduced']
        
        return X

class MakeModelComposite(BaseEstimator, TransformerMixin):
    """Create make_model composite feature."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if 'make' in X.columns and 'model' in X.columns:
            X['make_model'] = X['make'] + '_' + X['model']
            
            # Simple target encoding (mean target by make_model)
            if hasattr(self, 'target_means_'):
                X['make_model__te'] = X['make_model'].map(self.target_means_).fillna(0)
            else:
                X['make_model__te'] = 0
        
        return X

class DropVehicleId(BaseEstimator, TransformerMixin):
    """Drop vehicle ID column."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'vehicleID' in X.columns:
            X = X.drop(columns=['vehicleID'])
        return X

class DropInvalidYearIntroduced(BaseEstimator, TransformerMixin):
    """Drop rows with invalid yearIntroduced."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if 'yearIntroduced' in X.columns and 'saleDate' in X.columns:
            X['saleDate'] = pd.to_datetime(X['saleDate'], errors='coerce')
            X['sale_year'] = X['saleDate'].dt.year
            
            # Drop rows where yearIntroduced > sale_year
            invalid_mask = X['yearIntroduced'] > X['sale_year']
            if invalid_mask.any():
                print(f"Dropping {invalid_mask.sum()} rows with invalid yearIntroduced")
                X = X[~invalid_mask]
        
        return X

class CleanSaleDate(BaseEstimator, TransformerMixin):
    """Clean sale date column."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if 'saleDate' in X.columns:
            X['saleDate'] = pd.to_datetime(X['saleDate'], errors='coerce')
        
        return X

class CleanDoorNumber(BaseEstimator, TransformerMixin):
    """Clean door number column."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if 'doorNumber' in X.columns:
            # Convert to numeric, handle any string values
            X['doorNumber'] = pd.to_numeric(X['doorNumber'], errors='coerce')
        
        return X
