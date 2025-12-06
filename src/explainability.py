"""
Model explainability utilities using SHAP values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from typing import Optional, Any


class SHAPExplainer:
    """Wrapper for SHAP-based model explainability."""
    
    def __init__(self, model: Any, X_background: pd.DataFrame):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            X_background: Background data for SHAP
        """
        self.model = model
        self.X_background = X_background
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self) -> None:
        """Create SHAP explainer based on model type."""
        self.explainer = shap.Explainer(self.model, self.X_background)
    
    def calculate_shap_values(self, X: pd.DataFrame) -> Any:
        """
        Calculate SHAP values for given data.
        
        Args:
            X: Features to explain
            
        Returns:
            SHAP values object
        """
        if self.explainer is None:
            self.create_explainer()
        
        self.shap_values = self.explainer(X)
        return self.shap_values
    
    def get_feature_importance(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Get feature importance based on mean absolute SHAP values.
        
        Args:
            X: Features (if None, use cached shap_values)
            
        Returns:
            DataFrame with feature importance
        """
        if X is not None:
            self.calculate_shap_values(X)
        
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Run calculate_shap_values() first.")
        
        shap_df = pd.DataFrame(self.shap_values.values, columns=self.X_background.columns)
        importance = abs(shap_df).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': importance.index,
            'mean_abs_shap': importance.values
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        
        importance_df['cumsum'] = importance_df['mean_abs_shap'].cumsum()
        importance_df['cumsum_pct'] = (
            importance_df['cumsum'] / importance_df['mean_abs_shap'].sum()
        )
        
        return importance_df
    
    def plot_bar(self, X: Optional[pd.DataFrame] = None, 
                max_display: int = 15, save_path: Optional[str] = None) -> None:
        """
        Create SHAP bar plot showing feature importance.
        
        Args:
            X: Features (if None, use cached shap_values)
            max_display: Maximum features to display
            save_path: Optional path to save figure
        """
        if X is not None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.plots.bar(self.shap_values, max_display=max_display)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()
    
    def plot_beeswarm(self, X: Optional[pd.DataFrame] = None, 
                     max_display: int = 15, save_path: Optional[str] = None) -> None:
        """
        Create SHAP beeswarm plot showing feature effects.
        
        Args:
            X: Features (if None, use cached shap_values)
            max_display: Maximum features to display
            save_path: Optional path to save figure
        """
        if X is not None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(self.shap_values, max_display=max_display)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()
    
    def plot_waterfall(self, instance_index: int = 0, 
                      save_path: Optional[str] = None) -> None:
        """
        Create waterfall plot for a single prediction.
        
        Args:
            instance_index: Index of instance to explain
            save_path: Optional path to save figure
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Run calculate_shap_values() first.")
        
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(self.shap_values[instance_index])
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()
    
    def explain_instance(self, X_instance: pd.DataFrame, 
                        n_top_features: int = 5) -> pd.DataFrame:
        """
        Get top contributing features for a single instance.
        
        Args:
            X_instance: Single instance to explain (1 row DataFrame)
            n_top_features: Number of top features to return
            
        Returns:
            DataFrame with top features and their contributions
        """
        shap_vals = self.calculate_shap_values(X_instance)
        
        feature_impacts = pd.DataFrame({
            'feature': X_instance.columns,
            'feature_value': X_instance.iloc[0].values,
            'shap_value': shap_vals.values[0]
        })
        
        feature_impacts['abs_shap'] = abs(feature_impacts['shap_value'])
        top_features = feature_impacts.nlargest(n_top_features, 'abs_shap')
        
        return top_features[['feature', 'feature_value', 'shap_value', 'abs_shap']]
    
    def save_explainer(self, path: str) -> None:
        """
        Save explainer to disk.
        
        Args:
            path: Path to save explainer
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.explainer, f)
    
    @staticmethod
    def load_explainer(path: str) -> 'SHAPExplainer':
        """
        Load explainer from disk.
        
        Args:
            path: Path to explainer file
            
        Returns:
            SHAPExplainer instance
        """
        import pickle
        with open(path, 'rb') as f:
            explainer = pickle.load(f)
        
        # Create wrapper object
        shap_explainer = SHAPExplainer.__new__(SHAPExplainer)
        shap_explainer.explainer = explainer
        shap_explainer.model = None
        shap_explainer.X_background = None
        shap_explainer.shap_values = None
        
        return shap_explainer


def create_shap_report(model: Any, X_train: pd.DataFrame, 
                      output_dir: str, model_name: str) -> pd.DataFrame:
    """
    Create comprehensive SHAP analysis report.
    
    Args:
        model: Trained model
        X_train: Training features
        output_dir: Directory to save plots
        model_name: Model name for file naming
        
    Returns:
        Feature importance DataFrame
    """
    explainer = SHAPExplainer(model, X_train)
    explainer.calculate_shap_values(X_train)
    
    # Get feature importance
    importance = explainer.get_feature_importance()
    
    # Create plots
    explainer.plot_bar(
        max_display=15,
        save_path=f"{output_dir}/shap_bar_{model_name}.png"
    )
    
    explainer.plot_beeswarm(
        max_display=15,
        save_path=f"{output_dir}/shap_beeswarm_{model_name}.png"
    )
    
    # Save feature importance
    importance.to_csv(f"{output_dir}/feature_importance_{model_name}.csv", index=False)
    
    # Save explainer
    explainer.save_explainer(f"{output_dir}/explainer_{model_name}.pkl")
    
    return importance