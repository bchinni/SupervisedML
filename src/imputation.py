"""
Data imputation utilities using iterative imputation.
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from typing import Optional, Tuple


class DataImputer:
    """Wrapper for iterative imputation with decision tree estimator."""
    
    def __init__(self, max_iter: int = 100, random_state: int = 99, verbose: int = 2):
        """
        Initialize imputer.
        
        Args:
            max_iter: Maximum imputation iterations
            random_state: Random seed for reproducibility
            verbose: Verbosity level
        """
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.imputer = None
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """
        Fit imputer and transform data.
        
        Args:
            df: Input DataFrame with potential missing values
            
        Returns:
            Tuple of (imputed DataFrame, whether imputation was needed)
        """
        # Check if imputation is needed
        if not df.isna().any().any():
            print("No missing values – no imputation needed.")
            return df.copy(), False
        
        print("Imputation needed – starting IterativeImputer...")
        
        # Store column names and index
        columns = df.columns
        index = df.index
        
        # Initialize imputer
        self.imputer = IterativeImputer(
            estimator=DecisionTreeRegressor(max_features="sqrt", 
                                           random_state=self.random_state),
            missing_values=np.nan,
            max_iter=self.max_iter,
            verbose=self.verbose
        )
        
        # Fit and transform
        imputed_array = self.imputer.fit_transform(df)
        imputed_df = pd.DataFrame(imputed_array, columns=columns, index=index)
        
        return imputed_df, True
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted imputer.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Imputed DataFrame
        """
        if self.imputer is None:
            raise ValueError("Imputer must be fitted first. Call fit_transform().")
        
        if not df.isna().any().any():
            return df.copy()
        
        columns = df.columns
        index = df.index
        
        imputed_array = self.imputer.transform(df)
        imputed_df = pd.DataFrame(imputed_array, columns=columns, index=index)
        
        return imputed_df


def impute_data(df: pd.DataFrame, max_iter: int = 100, 
                random_state: int = 99, verbose: int = 2) -> pd.DataFrame:
    """
    Convenience function for one-step imputation.
    
    Args:
        df: Input DataFrame
        max_iter: Maximum imputation iterations
        random_state: Random seed
        verbose: Verbosity level
        
    Returns:
        Imputed DataFrame
    """
    imputer = DataImputer(max_iter=max_iter, 
                         random_state=random_state, 
                         verbose=verbose)
    imputed_df, _ = imputer.fit_transform(df)
    return imputed_df