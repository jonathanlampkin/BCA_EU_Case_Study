from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from unidecode import unidecode

from src.price_model.transformers import (
    DateAndMileage,
    MakeModelComposite,
    DropVehicleId,
    YearChronologyGuard,
    CleanSaleDate,
    CleanDoorNumber,
)
from src.price_model.num_sanitize import NumericSanitizer, ColCfg


@dataclass
class DomainConfig:
    """User config settings to apply domain and modeling knowledge."""
    # luxury_brands: List[str] = field(default_factory=lambda: [
    #     "BMW", "Mercedes", "Audi", "Lexus", "Infiniti", "Acura"
    # ])


    numeric_cfg: Dict[str, ColCfg] = field(default_factory=lambda: {
        "kilometers": ColCfg(min_val=0, max_val=None, replace_with="median"),
        "cubeCapacity": ColCfg(min_val=None, max_val=None, replace_with="median"),
        "powerKW": ColCfg(min_val=0, max_val=None, replace_with="median"),
        "cylinder": ColCfg(min_val=2, max_val=16, replace_with="median"),
        "vehicle_age_years": ColCfg(min_val=0, max_val=None, replace_with="median"),
        "km_per_year": ColCfg(min_val=0, max_val=None, replace_with="median"),
        "years_since_intro_at_sale": ColCfg(min_val=0, max_val=None, replace_with="median"),
    })


    # Categorical column transformation configuration options:
    # {
    #   "<col>": {
    #       "ordinal": True|False,                    # if True, use ordinal_mapping and drop original
    #       "ordinal_mapping": {raw:str -> int},      # required when ordinal=True
    #       "mapped_name": "<new_col_name>",         # optional name of created ordinal column
    #       "bin_specs": { "<new_binary>": [values] },# create one or more binary columns, drop original
    #       "top_cum_q": 0.0..1.0 | None,             # None = skip OHE; else keep most frequent until q
    #       "high_card_threshold": int                 # if cardinality > threshold → target encode
    #   }
    # }
    categorical_cfg: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "colour": {"top_cum_q": 1.0}, # OHE with no bin into "Other"; hence top_cum_q = 1.0
        "aestheticGrade": {"ordinal": True, "ordinal_mapping": {"Very Bad": 1, "Bad": 2, "Medium": 3, "Good": 4, "Very Good": 5}},
        "mechanicalGrade": {"ordinal": True, "ordinal_mapping": {"Very Bad": 1, "Bad": 2, "Medium": 3, "Good": 4, "Very Good": 5}},
        "fuel": {"top_cum_q": 1.0}, 
        "transmission": {"top_cum_q": 1.0},
        "type": {"top_cum_q": 1.0},
        "make_model": {"top_cum_q": 0.70, "high_card_threshold": 50} # Only keeping top 70% of most frequent make/model combinations to avoid cardinality explosion, falls back to target encoding if cardinality exceeds 50.
    })
    # z-score normalization of numeric features (left to False so that preprocessed data is still in the same scale as the original data)
    zscore_normalize: bool = False


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocesses data using the DomainConfig class above with logical guardrails such as dedepulication, imputation, dropping all-NA or zero-variance columns etc.
    We use classes to clean the data so that we can use class methods to fit preprocessing to the train then transform the test to avoid data leakage.
    """

    def __init__(self, domain: Optional[DomainConfig] = None):
        self.domain = domain or DomainConfig()
        self._num_sanitizer: Optional[NumericSanitizer] = None
        self._norm_stats: Dict[str, Dict[str, float]] = {}
        # removed once-only logging flag for dropped invalid rows

    def fit(self, X: pd.DataFrame, y=None):
        X = self._apply_stateless_transforms(X)
        # Use numeric_cfg as-is
        base_cfg_present = {k: v for k, v in self.domain.numeric_cfg.items() if k in X.columns}
        self._num_sanitizer = NumericSanitizer(base_cfg_present).fit(X)
        Xs = self._num_sanitizer.transform(X.copy())
        num_cols = Xs.select_dtypes(include=[np.number]).columns
        self._norm_stats = {c: {"mean": float(Xs[c].mean()), "std": float(Xs[c].std() or 1.0)} for c in num_cols}
        return self

    def transform(self, X: pd.DataFrame):
        X = self._apply_stateless_transforms(X.copy())
        if self._num_sanitizer is None:
            base_cfg_present = {k: v for k, v in self.domain.numeric_cfg.items() if k in X.columns}
            self._num_sanitizer = NumericSanitizer(base_cfg_present).fit(X)
        X = self._num_sanitizer.transform(X)
        # Recompute km_per_year AFTER kilometers sanitization
        if "kilometers" in X.columns and "vehicle_age_years" in X.columns:
            X["km_per_year"] = X["kilometers"] / X["vehicle_age_years"].replace(0, np.nan)
        # Categorical encoding
        X = self._apply_categorical_config(X)
        X = self._finalize(X)
        if self.domain.zscore_normalize:
            X = self._zscore_normalize(X)
        n_rows, n_cols = X.shape
        if n_cols > max(1, n_rows // 2):
            print(f"⚠️  Warning: feature count ({n_cols}) is large relative to rows ({n_rows}). Consider tightening categorical_cfg (top_cum_q/bin_specs/ordinal) or using target encoding.")
        return X

    def fit_transform(self, X: pd.DataFrame, y=None):
        return self.fit(X, y).transform(X)

    def _apply_stateless_transforms(self, X: pd.DataFrame) -> pd.DataFrame:
        X = CleanSaleDate().fit_transform(X)
        X = X.drop_duplicates()
        X = CleanDoorNumber().fit_transform(X)
        before = len(X)
        X = YearChronologyGuard().fit_transform(X)
        dropped = before - len(X)
        if dropped > 0:
            print(f"Dropped {dropped} rows for invalid chronology (yearIntroduced > sale_year)")
        if "cubeCapacity" in X.columns:
            cc = pd.to_numeric(X["cubeCapacity"], errors="coerce")
            cc = cc.where(cc > 0)
            X["cubeCapacity"] = cc
        X = DateAndMileage().fit_transform(X)
        # Year chronology and derived fields are handled in YearChronologyGuard
        # Build composite then drop original make/model to avoid duplicate OHE
        X = MakeModelComposite().fit_transform(X)
        if 'make' in X.columns:
            X = X.drop(columns=['make'])
        if 'model' in X.columns:
            X = X.drop(columns=['model'])
        X = DropVehicleId().fit_transform(X)
        # Clean categorical text to avoid excess cardinality
        obj_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(obj_cols) > 0:
            def _clean_ascii(s):
                if pd.isna(s):
                    return s
                s = unidecode(str(s))
                s = s.lower()
                s = re.sub(r"[^a-z0-9 ]+", " ", s)
                return re.sub(r"\\s+", " ", s).strip()
            X[obj_cols] = X[obj_cols].apply(lambda col: col.map(_clean_ascii))
        return X


    def _apply_categorical_config(self, X: pd.DataFrame) -> pd.DataFrame:
        cat_cfg = self.domain.categorical_cfg or {}
        # Exclude date columns from categorical processing
        date_cols = ['saleDate', 'registrationDate']
        cat_cols = [col for col in X.select_dtypes(include=['object', 'category']).columns.tolist() 
                   if col not in date_cols]
        for col in cat_cols:
            if col == "doorNumber":
                X[col] = pd.to_numeric(X[col], errors='coerce')
                continue
            s = X[col].astype('object')
            cfg = cat_cfg.get(col, {})
            processed = False
            if cfg.get("ordinal", False) and isinstance(cfg.get("ordinal_mapping"), dict):
                X[cfg.get("mapped_name", f"{col}_ordinal")] = s.map(cfg["ordinal_mapping"])
                processed = True
            else:
                bin_specs = cfg.get("bin_specs", {}) or {}
                for new_col, values in bin_specs.items():
                    X[new_col] = s.isin(values).astype(int)
                    processed = True
                cardinality = int(s.nunique(dropna=True))
                high_card_threshold = int(cfg.get("high_card_threshold", 50))
                if cardinality > high_card_threshold and 'targetPrice' in X.columns and 'saleDate' in X.columns:
                    print(f"{col}: cardinality={cardinality} > {high_card_threshold}. Using time-series target encoding (no leakage). top_q={cfg.get('top_cum_q', 1.0)}")
                    X[f"{col}__te"] = self._timeseries_target_encode(X, col, y_col='targetPrice', date_col='saleDate')
                    processed = True
                else:
                    top_q = cfg.get("top_cum_q", 1.0)
                    if top_q is not None:
                        vc = s.value_counts(normalize=True, dropna=False).sort_values(ascending=False)
                        cum = vc.cumsum()
                        levels = list(vc.index[cum <= float(top_q)])
                        if not levels and len(vc) > 0:
                            levels = [vc.index[0]]
                        # Drop one baseline to avoid perfect multicollinearity (k-1). Prefer first non-NaN level.
                        non_nan_levels = [lvl for lvl in vc.index if not (isinstance(lvl, float) and np.isnan(lvl))]
                        baseline = non_nan_levels[0] if non_nan_levels else vc.index[0]
                        keep_levels = [lvl for lvl in levels if lvl != baseline]
                        print(f"{col}: OHE top_q={top_q} -> kept {len(keep_levels)} of {cardinality} (k-1)")
                        for lvl in keep_levels:
                            col_name = f"{col}__{lvl}"
                            X[col_name] = (s == lvl).astype(int)
                        processed = True
            if processed:
                X = X.drop(columns=[col])
        return X

    def _timeseries_target_encode(self, X: pd.DataFrame, col: str, y_col: str = 'targetPrice', date_col: str = 'saleDate') -> pd.Series:
        df = X[[col, y_col, date_col]].copy()
        if df[date_col].dtype == 'object' and hasattr(df[date_col].iloc[0], 'year'):
            df[date_col] = pd.to_datetime(df[date_col])
        else:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(date_col)
        # Compute cumulative (past-only) means to avoid leakage
        grp = df.groupby(col, sort=False)
        counts = grp.cumcount() 
        csum = grp[y_col].cumsum() - df[y_col]
        ccount = counts
        # Overall prior using only rows strictly before current: sum/count of observed y up to previous row
        y_nonnull = df[y_col].copy()
        y_nonnull_mask = y_nonnull.notna().astype(int)
        overall_sum_prior = y_nonnull.fillna(0).cumsum() - y_nonnull.fillna(0)
        overall_cnt_prior = y_nonnull_mask.cumsum() - y_nonnull_mask
        prior = np.where(overall_cnt_prior > 0, overall_sum_prior / overall_cnt_prior, 0.0)
        # Smoothing towards overall prior with small weight alpha
        alpha = 10.0
        enc = np.empty(len(df), dtype=float)
        # First occurrence (no past) -> fixed prior 0.0
        mask_first = (ccount == 0)
        enc[mask_first] = 0.0
        # Subsequent occurrences -> smoothed past mean
        mask_rest = ~mask_first
        enc[mask_rest] = (csum[mask_rest] + alpha * prior[mask_rest]) / (ccount[mask_rest] + alpha)
        enc = pd.Series(enc, index=df.index)
        enc = enc.sort_index()
        return enc

    def _finalize(self, X: pd.DataFrame) -> pd.DataFrame:
        # Drop columns that shouldn't be in final dataset
        for c in ["vehicleID", "registrationDate", "powerHP", "yearIntroduced"]:
            if c in X.columns:
                X = X.drop(columns=[c])
        
        # Remove duplicates and handle missing values
        X = X.loc[:, ~X.columns.duplicated()]
        num_cols = X.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            if X[c].isnull().any() or np.isinf(X[c]).any():
                col = pd.to_numeric(X[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
                X[c] = col.fillna(col.median())
        
        # Remove all-NA and zero-variance columns
        all_na = [c for c in X.columns if X[c].isna().all()]
        if all_na:
            X = X.drop(columns=all_na)
        nunique = X.nunique(dropna=False)
        zero_var = [c for c, k in nunique.items() if k <= 1]
        if zero_var:
            X = X.drop(columns=zero_var)
        return X

    def _zscore_normalize(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._norm_stats:
            return X
        for c, stats in self._norm_stats.items():
            if c in X.columns:
                std = stats.get("std", 1.0) or 1.0
                X[c] = (X[c] - stats.get("mean", 0.0)) / std
        return X


def main():
    """Main entry point for preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess car price data')
    parser.add_argument('--input_path', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output_path', type=str, required=True, help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_path}")
    df = pd.read_csv(args.input_path)
    print(f"Original data shape: {df.shape}")
    
    # Apply preprocessing
    preprocessor = Preprocessor(DomainConfig())
    df_processed = preprocessor.fit_transform(df)
    print(f"Processed data shape: {df_processed.shape}")
    
    # Save processed data
    df_processed.to_csv(args.output_path, index=False)
    print(f"Processed data saved to {args.output_path}")


if __name__ == "__main__":
    main()
