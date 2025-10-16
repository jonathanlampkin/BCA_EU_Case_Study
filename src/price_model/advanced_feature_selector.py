import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureSelector:
    """Advanced feature selection combining multiple methods."""
    
    def __init__(self, correlation_threshold: float = 0.80, vif_threshold: float = 10.0):
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.selected_features = []
        self.feature_scores = {}
        self.removal_log = []
        
    def remove_multicollinearity(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove multicollinear features using correlation and VIF."""
        print("Removing multicollinear features...")
        
        # Step 1: Remove high correlation pairs
        X_clean = self._remove_high_correlation(X)
        
        # Step 2: Remove high VIF features
        X_clean = self._remove_high_vif(X_clean)
        
        print(f"Features removed: {len(X.columns) - len(X_clean.columns)}")
        print(f"Remaining features: {len(X_clean.columns)}")
        
        return X_clean
    
    def _remove_high_correlation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features with high correlation."""
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.correlation_threshold)]
        
        for col in to_drop:
            self.removal_log.append(f"Correlation: {col} (threshold: {self.correlation_threshold})")
        
        return X.drop(columns=to_drop)
    
    def _remove_high_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features with high VIF iteratively."""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        X_temp = X.copy()
        iteration = 1
        
        while True:
            # Calculate VIF
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X_temp.columns
            vif_data["VIF"] = [variance_inflation_factor(X_temp.values, i) for i in range(X_temp.shape[1])]
            
            max_vif = vif_data["VIF"].max()
            
            if max_vif > self.vif_threshold:
                feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
                self.removal_log.append(f"VIF Iteration {iteration}: {feature_to_remove} (VIF: {max_vif:.2f})")
                X_temp = X_temp.drop(columns=[feature_to_remove])
                iteration += 1
            else:
                break
            
            if iteration > 50:  # Safety check
                print("VIF removal stopped after 50 iterations")
                break
        
        return X_temp
    
    def compute_feature_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """Compute feature scores using multiple methods."""
        print("Computing feature scores using multiple methods...")
        
        scores = {}
        
        # 1. Statistical scores (F-test)
        f_scores, _ = f_regression(X, y)
        scores['f_test'] = dict(zip(X.columns, f_scores))
        
        # 2. Mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        scores['mutual_info'] = dict(zip(X.columns, mi_scores))
        
        # 3. Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        scores['random_forest'] = dict(zip(X.columns, rf.feature_importances_))
        
        # 4. Lasso coefficients
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X, y)
        scores['lasso'] = dict(zip(X.columns, np.abs(lasso.coef_)))
        
        self.feature_scores = scores
        return scores
    
    def select_features_by_performance(self, X: pd.DataFrame, y: pd.Series, 
                                     model, min_features: int = 5, max_features: int = 20) -> List[str]:
        """Select features based on model performance."""
        print(f"Selecting features based on performance (min: {min_features}, max: {max_features})...")
        
        # Start with all features
        current_features = list(X.columns)
        best_score = -np.inf
        best_features = current_features.copy()
        
        # Forward selection
        while len(current_features) > min_features:
            scores = []
            
            for feature in current_features:
                # Remove feature and test performance
                temp_features = [f for f in current_features if f != feature]
                if len(temp_features) < min_features:
                    continue
                
                X_temp = X[temp_features]
                score = cross_val_score(model, X_temp, y, cv=5, scoring='neg_mean_squared_error').mean()
                scores.append((feature, score))
            
            if not scores:
                break
            
            # Find feature that improves score when removed
            scores.sort(key=lambda x: x[1], reverse=True)
            worst_feature, worst_score = scores[0]
            
            if worst_score > best_score:
                best_score = worst_score
                best_features = [f for f in current_features if f != worst_feature]
                current_features = best_features.copy()
                print(f"Removed {worst_feature}, score: {worst_score:.4f}")
            else:
                break
        
        return best_features
    
    def ensemble_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                 n_features: int = 15) -> List[str]:
        """Ensemble feature selection combining multiple methods."""
        print(f"Performing ensemble feature selection for {n_features} features...")
        
        # Compute scores
        scores = self.compute_feature_scores(X, y)
        
        # Normalize scores
        normalized_scores = {}
        for method, method_scores in scores.items():
            max_score = max(method_scores.values())
            normalized_scores[method] = {k: v/max_score for k, v in method_scores.items()}
        
        # Combine scores (equal weight)
        combined_scores = {}
        for feature in X.columns:
            combined_scores[feature] = np.mean([normalized_scores[method][feature] 
                                              for method in normalized_scores.keys()])
        
        # Select top features
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:n_features]]
        
        print(f"Selected features: {selected_features}")
        
        return selected_features
    
    def comprehensive_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                      model, target_features: int = 15) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Comprehensive feature selection pipeline."""
        print("Starting comprehensive feature selection...")
        
        # Step 1: Remove multicollinearity
        X_clean = self.remove_multicollinearity(X)
        
        # Step 2: Ensemble selection
        selected_features = self.ensemble_feature_selection(X_clean, y, target_features)
        
        # Step 3: Performance-based refinement
        X_selected = X_clean[selected_features]
        final_features = self.select_features_by_performance(X_selected, y, model)
        
        # Final dataset
        X_final = X_clean[final_features]
        
        # Create summary
        summary = {
            'initial_features': len(X.columns),
            'after_multicollinearity': len(X_clean.columns),
            'after_ensemble_selection': len(selected_features),
            'final_features': len(final_features),
            'removed_features': list(set(X.columns) - set(final_features)),
            'selected_features': final_features,
            'removal_log': self.removal_log,
            'feature_scores': self.feature_scores
        }
        
        print(f"Feature selection complete:")
        print(f"  Initial: {summary['initial_features']} features")
        print(f"  Final: {summary['final_features']} features")
        print(f"  Removed: {len(summary['removed_features'])} features")
        
        return X_final, summary
