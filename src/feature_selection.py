"""
Feature selection utilities using BorutaSHAP.
"""

import pandas as pd
import numpy as np
from BorutaShap import BorutaShap
from xgboost import XGBClassifier
from typing import List, Optional


class BorutaFeatureSelector:
    """Wrapper for BorutaShap feature selection."""
    
    def __init__(self, percentile: int = 90, pvalue: float = 0.05, 
                 random_state: int = 42):
        """
        Initialize feature selector.
        
        Args:
            percentile: Percentile for feature importance threshold
            pvalue: P-value threshold for feature selection
            random_state: Random seed for reproducibility
        """
        self.percentile = percentile
        self.pvalue = pvalue
        self.random_state = random_state
        self.selector = None
        self.selected_features = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Fit feature selector and return selected features.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            List of selected feature names
        """
        model = XGBClassifier(random_state=self.random_state)
        
        self.selector = BorutaShap(
            model=model, 
            importance_measure='shap', 
            classification=True,
            percentile=self.percentile,
            pvalue=self.pvalue
        )
        
        # Fit the model
        self.selector.fit(X, y)
        
        # Get selected features
        self.selected_features = self.selector.Subset().columns.tolist()
        
        print(f"Selected {len(self.selected_features)} features out of {len(X.columns)}")
        
        return self.selected_features
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame to include only selected features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with selected features only
        """
        if self.selected_features is None:
            raise ValueError("Selector must be fitted first. Call fit().")
        
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit selector and transform in one step.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            DataFrame with selected features only
        """
        self.fit(X, y)
        return self.transform(X)


def run_feature_selection_sweep(X_train: pd.DataFrame, y_train: pd.Series,
                                percentile_range: range = range(50, 100, 5),
                                pvalue: float = 0.05,
                                random_state: int = 42) -> dict:
    """
    Run feature selection across multiple percentile values.
    
    Args:
        X_train: Training features
        y_train: Training target
        percentile_range: Range of percentiles to test
        pvalue: P-value threshold
        random_state: Random seed
        
    Returns:
        Dictionary mapping percentile -> selected features
    """
    results = {}
    
    for perc in percentile_range:
        print(f"\n{'='*60}")
        print(f"Running Boruta with percentile={perc}")
        print(f"{'='*60}")
        
        selector = BorutaFeatureSelector(
            percentile=perc,
            pvalue=pvalue,
            random_state=random_state
        )
        
        selected = selector.fit(X_train, y_train)
        results[perc] = selected
        
    return results