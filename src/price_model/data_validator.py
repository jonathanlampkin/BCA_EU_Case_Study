"""
Simple data validation script.
"""
import pandas as pd
import numpy as np
import json
import os
import argparse
from datetime import datetime

def validate_data(data_path: str, artifacts_dir: str):
    """Validate data quality and create validation report."""
    print("Validating data quality...")
    
    # Create artifacts directory if it doesn't exist
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Basic validation
    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "data_path": data_path,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "target_stats": {
            "min": float(df['targetPrice'].min()),
            "max": float(df['targetPrice'].max()),
            "mean": float(df['targetPrice'].mean()),
            "std": float(df['targetPrice'].std()),
            "skew": float(df['targetPrice'].skew())
        } if 'targetPrice' in df.columns else None
    }
    
    # Save validation report
    report_path = os.path.join(artifacts_dir, "data_validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"Data validation complete! Report saved to {report_path}")
    print(f"Data shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    if 'targetPrice' in df.columns:
        print(f"Target range: {df['targetPrice'].min():.0f} - {df['targetPrice'].max():.0f}")
        print(f"Target skew: {df['targetPrice'].skew():.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate data quality")
    parser.add_argument("--data", type=str, required=True, help="Path to data file")
    parser.add_argument("--artifacts", type=str, required=True, help="Path to artifacts directory")
    
    args = parser.parse_args()
    validate_data(args.data, args.artifacts)
