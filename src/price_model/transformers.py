"""
Column Transformations for the preprocessor.
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
            X['saleDate'] = pd.to_datetime(X['saleDate'], errors='coerce', dayfirst=True)
            X['sale_year'] = X['saleDate'].dt.year
            X['sale_month'] = X['saleDate'].dt.month
        
        # Calculate vehicle age
        if 'registrationDate' in X.columns and 'saleDate' in X.columns:
            X['registrationDate'] = pd.to_datetime(X['registrationDate'], errors='coerce', dayfirst=True)
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

class YearChronologyGuard(BaseEstimator, TransformerMixin):
    """Ensure chronological consistency and derive years_since_intro_at_sale.

    - Coerce saleDate to datetime and create sale_year
    - Drop rows with yearIntroduced > sale_year
    - Treat implausible yearIntroduced (<1900) as missing
    - Derive years_since_intro_at_sale = sale_year - yearIntroduced (with NaNs handled)
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if 'saleDate' in X.columns:
            X['saleDate'] = pd.to_datetime(X['saleDate'], errors='coerce', dayfirst=True)
            X['sale_year'] = X['saleDate'].dt.year
        
        if 'yearIntroduced' in X.columns and 'sale_year' in X.columns:
            yintro = pd.to_numeric(X['yearIntroduced'], errors='coerce')
            sale_year = pd.to_numeric(X['sale_year'], errors='coerce')
            # Drop future-introduced rows
            invalid_future = yintro > sale_year
            if invalid_future.any():
                X = X.loc[~invalid_future].copy()
                yintro = yintro.loc[X.index]
                sale_year = sale_year.loc[X.index]
            # Mask implausible historical years
            yintro = yintro.mask(yintro < 1900)
            X['yearIntroduced'] = yintro
            # Derive years since intro
            X['years_since_intro_at_sale'] = (sale_year - yintro).astype(float)
        
        return X

class CleanSaleDate(BaseEstimator, TransformerMixin):
    """Clean sale date column."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if 'saleDate' in X.columns:
            X['saleDate'] = pd.to_datetime(X['saleDate'], errors='coerce', dayfirst=True)
        
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
