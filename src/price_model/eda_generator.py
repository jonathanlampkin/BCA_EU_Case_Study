#!/usr/bin/env python3
"""
Simple EDA Report Generator
Generates a quick EDA report without the heavy ydata_profiling overhead.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from ydata_profiling import ProfileReport

def generate_quick_eda_report(data_path: str, output_dir: str = "artifacts"):
    """Generate a quick EDA report using basic pandas and matplotlib."""
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    print(f"Data shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic info
    print("\n=== BASIC INFO ===")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Data types
    print("\n=== DATA TYPES ===")
    print(df.dtypes.value_counts())
    
    # Missing values
    print("\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    }).sort_values('Missing Count', ascending=False)
    
    print(missing_df[missing_df['Missing Count'] > 0])
    
    # Numeric columns summary
    print("\n=== NUMERIC COLUMNS SUMMARY ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    
    # Categorical columns summary
    print("\n=== CATEGORICAL COLUMNS SUMMARY ===")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Most common: {df[col].value_counts().head(3).to_dict()}")
    
    # Generate simple visualizations
    print("\n=== GENERATING VISUALIZATIONS ===")
    
    # 1. Missing values heatmap
    if df.isnull().sum().sum() > 0:
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/missing_values_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Missing values heatmap saved")
    
    # 2. Numeric columns distribution
    if len(numeric_cols) > 0:
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 4 * n_rows))
        for i, col in enumerate(numeric_cols):
            plt.subplot(n_rows, n_cols, i + 1)
            df[col].hist(bins=30, alpha=0.7)
            plt.title(f'{col} Distribution')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/numeric_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Numeric distributions saved")
    
    # 3. Correlation heatmap for numeric columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Correlation heatmap saved")
    
    # 4. Categorical columns bar plots
    if len(categorical_cols) > 0:
        n_cols = min(3, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        for i, col in enumerate(categorical_cols):
            plt.subplot(n_rows, n_cols, i + 1)
            value_counts = df[col].value_counts().head(10)  # Top 10 values
            value_counts.plot(kind='bar')
            plt.title(f'{col} Value Counts (Top 10)')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/categorical_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Categorical distributions saved")
    
    # Save summary to text file
    summary_path = f"{output_dir}/eda_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"EDA Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
        
        f.write("MISSING VALUES:\n")
        f.write(missing_df.to_string())
        f.write("\n\n")
        
        f.write("DATA TYPES:\n")
        f.write(df.dtypes.value_counts().to_string())
        f.write("\n\n")
        
        if len(numeric_cols) > 0:
            f.write("NUMERIC SUMMARY:\n")
            f.write(df[numeric_cols].describe().to_string())
            f.write("\n\n")
    
    print(f"✓ EDA summary saved to {summary_path}")
    
    # Generate YData Profiling report
    print("  Generating YData Profiling report...")
    try:
        import warnings
        warnings.filterwarnings('ignore')
        
        report = ProfileReport(
            df, 
            title=f"Car Pricing Dataset EDA - {os.path.basename(data_path)}", 
            explorative=True,
            minimal=False,
            correlations={
                "pearson": {"calculate": True},
                "spearman": {"calculate": True},
                "kendall": {"calculate": True},
                "phi_k": {"calculate": True},
                "cramers": {"calculate": True}
            },
            interactions={
                "continuous": True,
                "targets": ["targetPrice"] if "targetPrice" in df.columns else None
            },
            missing_diagrams={
                "matrix": True,
                "bar": True,
                "heatmap": True,
                "dendrogram": True
            },
            duplicates={"head": 10},
            samples={"head": 5, "tail": 5}
        )
        
        # Save HTML report
        html_path = os.path.join(output_dir, f"ydata_report_{os.path.basename(data_path).replace('.csv', '')}.html")
        report.to_file(html_path)
        
        # Save JSON report
        json_path = os.path.join(output_dir, f"ydata_report_{os.path.basename(data_path).replace('.csv', '')}.json")
        with open(json_path, "w") as f:
            f.write(report.to_json())
        
        print(f"✓ YData Profiling report saved: {html_path}")
        
    except Exception as e:
        print(f"⚠️ YData Profiling failed: {e}")
        print("  Continuing with other visualizations...")
    
    print(f"\n=== EDA REPORT COMPLETE ===")
    print(f"All files saved to: {output_dir}/")
    print("Generated files:")
    print("  - eda_summary.txt")
    print("  - missing_values_heatmap.png")
    print("  - numeric_distributions.png") 
    print("  - correlation_heatmap.png")
    print("  - categorical_distributions.png")
    print("  - ydata_report_*.html (interactive report)")
    print("  - ydata_report_*.json (data export)")

if __name__ == "__main__":
    # Generate EDA for both raw and clean data
    print("Generating EDA report for raw data...")
    generate_quick_eda_report("data/EUDS_CaseStudy_Pricing.csv")
    
    if os.path.exists("artifacts/clean_data.csv"):
        print("\nGenerating EDA report for clean data...")
        generate_quick_eda_report("artifacts/clean_data.csv", "artifacts/clean_eda")
