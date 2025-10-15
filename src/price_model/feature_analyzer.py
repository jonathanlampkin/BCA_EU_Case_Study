"""
Feature importance analysis and SHAP value computation for model interpretability.
"""
import numpy as np
import pandas as pd
import matplotlib
# Use a non-interactive backend to avoid Tkinter/threading issues in headless/multiprocessing runs
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from typing import Dict, List, Any, Tuple, Optional
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class FeatureAnalyzer:
    """Comprehensive feature importance analysis with SHAP values.
    
    Defaults are tuned to be quiet and reasonably fast for iterative runs.
    """
    
    def __init__(self, model, feature_names: List[str], *, verbose: bool = False,
                 enable_shap_summary: bool = True,
                 enable_shap_waterfall: bool = False,
                 enable_interactions: bool = False,
                 shap_sample_size: int = 300):
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = None
        self.shap_values = None
        self.verbose = verbose
        self.enable_shap_summary = enable_shap_summary
        self.enable_shap_waterfall = enable_shap_waterfall
        self.enable_interactions = enable_interactions
        self.shap_sample_size = shap_sample_size
        
    def compute_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute feature importance from the trained model."""
        importance_dict = {}
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models (LightGBM, XGBoost, Random Forest)
            importances = self.model.feature_importances_
            importance_dict = dict(zip(self.feature_names, importances))
        elif hasattr(self.model, 'coef_'):
            # Linear models (Ridge, Lasso, ElasticNet)
            coefs = np.abs(self.model.coef_)
            importance_dict = dict(zip(self.feature_names, coefs))
        else:
            # Fallback: use permutation importance
            from sklearn.inspection import permutation_importance
            perm_importance = permutation_importance(self.model, X, y, n_repeats=10, random_state=42)
            importance_dict = dict(zip(self.feature_names, perm_importance.importances_mean))
        
        return importance_dict
    
    def compute_shap_values(self, X: pd.DataFrame, sample_size: Optional[int] = None) -> np.ndarray:
        """Compute SHAP values for model interpretability.

        Performance safeguards:
        - Subsample to sample_size (default 1000)
        - Prefer TreeExplainer for tree models
        - Use small, independent masker/background
        - Disable additivity check for speed
        """
        if self.verbose:
            print("Computing SHAP values...")
        
        # Sample data for faster computation
        eff_sample = sample_size if sample_size is not None else self.shap_sample_size
        if len(X) > eff_sample:
            X_sample = X.sample(n=eff_sample, random_state=42)
        else:
            X_sample = X
        
        # Create SHAP explainer based on model type
        model_name = str(type(self.model)).lower()
        if ('lightgbm' in model_name) or ('xgboost' in model_name) or ('randomforest' in model_name):
            # Tree models: use fast TreeExplainer with small independent masker
            try:
                masker = shap.maskers.Independent(shap.sample(X_sample, min(500, len(X_sample)), random_state=42))
            except Exception:
                masker = shap.maskers.Independent(X_sample.iloc[: min(500, len(X_sample))])
            self.shap_explainer = shap.TreeExplainer(self.model, data=masker, feature_perturbation="interventional")
            # Compute SHAP values; disable additivity check for speed
            try:
                self.shap_values = self.shap_explainer.shap_values(X_sample, check_additivity=False)
            except TypeError:
                # Older SHAP versions may not support check_additivity here
                self.shap_values = self.shap_explainer.shap_values(X_sample)
        else:
            # Fallback: KernelExplainer (slow). Use a very small background.
            background = X_sample.iloc[: min(100, len(X_sample))]
            self.shap_explainer = shap.KernelExplainer(self.model.predict, background)
            # KernelExplainer returns list/array depending on SHAP version; keep as-is
            self.shap_values = self.shap_explainer.shap_values(X_sample)
        
        return self.shap_values
    
    def create_feature_importance_plot(self, importance_dict: Dict[str, float], 
                                     top_n: int = 15, save_path: str = "artifacts/feature_importance.png"):
        """Create feature importance visualization."""
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Top Feature Importances')
        plt.gca().invert_yaxis()
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Feature importance plot saved to {save_path}")
    
    def create_shap_summary_plot(self, X: pd.DataFrame, save_path: str = "artifacts/shap_summary.png"):
        """Create SHAP summary plot."""
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Ensure X and shap_values have the same number of rows
        if len(X) != len(self.shap_values):
            if self.verbose:
                print(f"Warning: X has {len(X)} rows but SHAP values have {len(self.shap_values)} rows. Using subset.")
            min_rows = min(len(X), len(self.shap_values))
            X_subset = X.iloc[:min_rows]
            shap_subset = self.shap_values[:min_rows]
        else:
            X_subset = X
            shap_subset = self.shap_values
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_subset, X_subset, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"SHAP summary plot saved to {save_path}")
    
    def create_shap_waterfall_plot(self, X: pd.DataFrame, instance_idx: int = 0, 
                                 save_path: str = "artifacts/shap_waterfall.png"):
        """Create SHAP waterfall plot for a specific instance."""
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Get prediction for the instance
        prediction = self.model.predict(X.iloc[[instance_idx]])[0]
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        try:
            # Use the correct parameter order for waterfall_plot
            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values[instance_idx],
                    base_values=self.shap_explainer.expected_value,
                    data=X.iloc[instance_idx],
                    feature_names=self.feature_names
                )
            )
            plt.title(f'SHAP Waterfall Plot - Instance {instance_idx} (Prediction: {prediction:.2f})')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception:
            # Silently skip if waterfall cannot be created on this backend/version
            pass
            plt.close()
        
        if self.verbose:
            print(f"SHAP waterfall plot saved to {save_path}")
    
    def analyze_feature_interactions(self, X: pd.DataFrame, top_features: int = 10) -> pd.DataFrame:
        """Analyze feature interactions using SHAP values."""
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Get top features by mean absolute SHAP value
        mean_shap = np.mean(np.abs(self.shap_values), axis=0)
        available_features = min(top_features, len(mean_shap))
        top_feature_indices = np.argsort(mean_shap)[-available_features:]
        
        # Compute interaction matrix
        interaction_matrix = np.zeros((available_features, available_features))
        feature_names = [self.feature_names[i] for i in top_feature_indices]
        
        for i, idx1 in enumerate(top_feature_indices):
            for j, idx2 in enumerate(top_feature_indices):
                if i != j:
                    # Compute correlation between SHAP values
                    corr = np.corrcoef(self.shap_values[:, idx1], self.shap_values[:, idx2])[0, 1]
                    interaction_matrix[i, j] = corr
        
        # Create DataFrame
        interaction_df = pd.DataFrame(interaction_matrix, 
                                    index=feature_names, 
                                    columns=feature_names)
        
        return interaction_df
    
    def create_feature_interaction_heatmap(self, interaction_df: pd.DataFrame, 
                                         save_path: str = "artifacts/feature_interactions.png"):
        """Create feature interaction heatmap."""
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(interaction_df, dtype=bool))
        sns.heatmap(interaction_df, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Interaction Matrix (SHAP-based)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Feature interaction heatmap saved to {save_path}")
    
    def get_feature_importance_ranking(self, importance_dict: Dict[str, float]) -> pd.DataFrame:
        """Get ranked feature importance DataFrame."""
        df = pd.DataFrame(list(importance_dict.items()), 
                         columns=['Feature', 'Importance'])
        df = df.sort_values('Importance', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        df['Importance_Percent'] = (df['Importance'] / df['Importance'].sum()) * 100
        
        return df
    
    def compare_feature_selection_methods(self, X: pd.DataFrame, y: pd.Series, 
                                        n_features: int = 10) -> Dict[str, List[str]]:
        """Compare different feature selection methods."""
        results = {}
        
        # 1. Model-based feature importance
        importance_dict = self.compute_feature_importance(X, y)
        model_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:n_features]
        results['Model_Importance'] = [f[0] for f in model_features]
        
        # 2. Recursive Feature Elimination
        rfe = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42), 
                 n_features_to_select=n_features)
        rfe.fit(X, y)
        results['RFE'] = [self.feature_names[i] for i in range(len(self.feature_names)) if rfe.support_[i]]
        
        # 3. Statistical feature selection
        selector = SelectKBest(score_func=f_regression, k=n_features)
        selector.fit(X, y)
        results['Statistical'] = [self.feature_names[i] for i in selector.get_support(indices=True)]
        
        # 4. SHAP-based selection
        if self.shap_values is None:
            self.compute_shap_values(X)
        mean_shap = np.mean(np.abs(self.shap_values), axis=0)
        shap_indices = np.argsort(mean_shap)[-n_features:]
        results['SHAP'] = [self.feature_names[i] for i in shap_indices]
        
        return results
    
    def generate_feature_analysis_report(self, X: pd.DataFrame, y: pd.Series, 
                                       output_dir: str = "artifacts") -> Dict[str, Any]:
        """Generate comprehensive feature analysis report."""
        if self.verbose:
            print("Generating comprehensive feature analysis report...")
        
        # Compute feature importance
        importance_dict = self.compute_feature_importance(X, y)
        
        # Compute SHAP values
        self.compute_shap_values(X)
        
        # Create visualizations
        self.create_feature_importance_plot(importance_dict, save_path=f"{output_dir}/feature_importance.png")
        if self.enable_shap_summary:
            self.create_shap_summary_plot(X, save_path=f"{output_dir}/shap_summary.png")
        if self.enable_shap_waterfall:
            self.create_shap_waterfall_plot(X, save_path=f"{output_dir}/shap_waterfall.png")
        
        # Feature interactions
        interaction_df = None
        if self.enable_interactions:
            interaction_df = self.analyze_feature_interactions(X)
            self.create_feature_interaction_heatmap(interaction_df, save_path=f"{output_dir}/feature_interactions.png")
        
        # Compare feature selection methods
        selection_comparison = self.compare_feature_selection_methods(X, y)
        
        # Generate ranking
        importance_ranking = self.get_feature_importance_ranking(importance_dict)
        
        # Save results
        importance_ranking.to_csv(f"{output_dir}/feature_importance_ranking.csv", index=False)
        
        # Create summary report
        report = {
            'feature_importance': importance_dict,
            'top_features': importance_ranking.head(10).to_dict('records'),
            'selection_comparison': selection_comparison,
            'shap_values_computed': self.shap_values is not None,
            'total_features': len(self.feature_names),
            'interaction_matrix': interaction_df.to_dict() if interaction_df is not None else {}
        }
        
        # Save report
        import json
        with open(f"{output_dir}/feature_analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        if self.verbose:
            print(f"Feature analysis report saved to {output_dir}/")
        
        return report
