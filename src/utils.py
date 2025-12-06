"""
General utility functions for I/O, configuration, and helpers.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Any, Optional, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def save_model(model: Any, filepath: str) -> None:
    """
    Save model to disk using pickle.
    
    Args:
        model: Model object to save
        filepath: Path to save model
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load model from disk.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model object
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def save_results(df: pd.DataFrame, filepath: str) -> None:
    """
    Save DataFrame to Excel or CSV based on extension.
    
    Args:
        df: DataFrame to save
        filepath: Output filepath
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.endswith('.xlsx'):
        df.to_excel(filepath, index=False)
    else:
        df.to_csv(filepath, index=False)
    
    print(f"Results saved to {filepath}")


def create_train_test_split(X: pd.DataFrame, y: pd.Series, 
                           test_size: float = 0.2,
                           random_state: int = 42,
                           stratify: bool = True) -> tuple:
    """
    Create stratified train-test split.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random seed
        stratify: Whether to stratify split
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    stratify_col = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_col, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


class DataScaler:
    """Wrapper for StandardScaler with categorical column handling."""
    
    def __init__(self, numerical_cols: list, categorical_cols: list):
        """
        Initialize scaler.
        
        Args:
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
        """
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.scaler = StandardScaler()
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler and transform data.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Scaled DataFrame
        """
        X_scaled = X.copy()
        
        if self.numerical_cols:
            X_num_scaled = self.scaler.fit_transform(X[self.numerical_cols])
            X_num_scaled = pd.DataFrame(
                X_num_scaled, 
                index=X.index, 
                columns=self.numerical_cols
            )
            
            if self.categorical_cols:
                X_scaled = pd.concat([X_num_scaled, X[self.categorical_cols]], axis=1)
            else:
                X_scaled = X_num_scaled
        
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scaler.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Scaled DataFrame
        """
        X_scaled = X.copy()
        
        if self.numerical_cols:
            X_num_scaled = self.scaler.transform(X[self.numerical_cols])
            X_num_scaled = pd.DataFrame(
                X_num_scaled, 
                index=X.index, 
                columns=self.numerical_cols
            )
            
            if self.categorical_cols:
                X_scaled = pd.concat([X_num_scaled, X[self.categorical_cols]], axis=1)
            else:
                X_scaled = X_num_scaled
        
        return X_scaled
    
    def save(self, filepath: str) -> None:
        """Save scaler to disk."""
        save_model(self, filepath)
    
    @staticmethod
    def load(filepath: str) -> 'DataScaler':
        """Load scaler from disk."""
        return load_model(filepath)


def setup_output_directory(base_dir: str = "output") -> Dict[str, str]:
    """
    Create output directory structure.
    
    Args:
        base_dir: Base output directory
        
    Returns:
        Dictionary with directory paths
    """
    dirs = {
        'base': base_dir,
        'models': f"{base_dir}/models",
        'results': f"{base_dir}/results",
        'plots': f"{base_dir}/plots",
        'shap': f"{base_dir}/shap"
    }
    
    for dir_path in dirs.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return dirs


def get_config() -> Dict[str, Any]:
    """
    Get default configuration parameters.
    
    Returns:
        Configuration dictionary
    """
    config = {
        'random_state': 42,
        'test_size': 0.2,
        'cv_folds': 5,
        'n_bayes_iter': 50,
        'missing_threshold': 0.4,
        'boruta_percentile': 90,
        'boruta_pvalue': 0.05,
        'max_impute_iter': 100,
        'n_permutations': 1000,
        'output_dir': 'output'
    }
    
    return config


def print_dataset_summary(X: pd.DataFrame, y: pd.Series, 
                         dataset_name: str = "Dataset") -> None:
    """
    Print summary statistics for dataset.
    
    Args:
        X: Features
        y: Target
        dataset_name: Name for display
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name} Summary")
    print(f"{'='*60}")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {len(X.columns)}")
    print(f"Outcome prevalence: {y.mean():.3f}")
    print(f"Positive cases: {y.sum()}")
    print(f"Negative cases: {len(y) - y.sum()}")
    print(f"{'='*60}\n")


def bootstrap_predictions(model: Any, X: pd.DataFrame, y: pd.Series, 
                         X_test: pd.DataFrame, y_test: pd.Series,
                         n_bootstraps: int = 1000, 
                         random_state: int = 42) -> Dict[str, float]:
    """
    Bootstrap model predictions for confidence intervals.
    
    Args:
        model: Trained model
        X: Training features
        y: Training target
        X_test: Test features
        y_test: Test target
        n_bootstraps: Number of bootstrap iterations
        random_state: Random seed
        
    Returns:
        Dictionary with mean, std, and confidence bounds
    """
    from sklearn.utils import resample
    
    np.random.seed(random_state)
    y_pred_boot_mean = []
    
    for _ in range(n_bootstraps):
        X_boot, y_boot = resample(X, y, replace=True)
        model.fit(X_boot, y_boot)
        y_pred_boot = model.predict_proba(X_test)[:, 1]
        y_pred_boot_mean.append(np.mean(y_pred_boot))
    
    mean_pred = np.mean(y_pred_boot_mean)
    std_pred = np.std(y_pred_boot_mean)
    std_error = std_pred / np.sqrt(n_bootstraps)
    
    return {
        'mean': mean_pred,
        'std': std_pred,
        'std_error': std_error,
        'lower_bound': mean_pred - (std_error * 1.96),
        'upper_bound': mean_pred + (std_error * 1.96)
    }