"""
Numeric data sanitization utilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ColCfg:
    """Configuration for column sanitization."""
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    replace_with: str = 'median'


class NumericSanitizer:
    """Sanitize numeric columns by handling outliers and missing values."""
    
    def __init__(self, col_configs: Dict[str, ColCfg] = None):
        self.col_configs = col_configs or {}
        self.stats_ = {}
    
    def fit(self, X: pd.DataFrame) -> 'NumericSanitizer':
        """Fit the sanitizer to the data."""
        for col in X.select_dtypes(include=[np.number]).columns:
            if col in self.col_configs:
                self.stats_[col] = {
                    'median': X[col].median(),
                    'mean': X[col].mean(),
                    'std': X[col].std()
                }
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by sanitizing numeric columns."""
        X_clean = X.copy()
        
        for col in X_clean.select_dtypes(include=[np.number]).columns:
            if col in self.col_configs:
                cfg = self.col_configs[col]
                
                # Handle outliers
                if cfg.min_val is not None:
                    X_clean.loc[X_clean[col] < cfg.min_val, col] = np.nan
                if cfg.max_val is not None:
                    X_clean.loc[X_clean[col] > cfg.max_val, col] = np.nan
                
                # Replace missing values
                if cfg.replace_with == 'median' and col in self.stats_:
                    X_clean[col] = X_clean[col].fillna(self.stats_[col]['median'])
                elif cfg.replace_with == 'mean' and col in self.stats_:
                    X_clean[col] = X_clean[col].fillna(self.stats_[col]['mean'])
        
        return X_clean
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(X).transform(X)
