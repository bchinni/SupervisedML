"""
Supervised Machine Learning Pipeline

A modular pipeline for binary classification with automated feature selection,
hyperparameter optimization, and model explainability.
"""

__version__ = "1.0.0"
__author__ = "bchinni"

from src.preprocessing import (
    clean_dataframe,
    classify_columns,
    analyze_missingness,
    drop_high_missing_vars
)

from src.imputation import DataImputer, impute_data

from src.feature_selection import BorutaFeatureSelector

from src.model_training import ModelTrainer

from src.evaluation import (
    ModelEvaluator,
    calculate_optimal_threshold,
    calculate_classification_metrics,
    plot_roc_curves
)

from src.explainability import SHAPExplainer, create_shap_report

from src.utils import (
    save_model,
    load_model,
    save_results,
    create_train_test_split,
    DataScaler,
    get_config
)

__all__ = [
    # Preprocessing
    'clean_dataframe',
    'classify_columns',
    'analyze_missingness',
    'drop_high_missing_vars',
    
    # Imputation
    'DataImputer',
    'impute_data',
    
    # Feature Selection
    'BorutaFeatureSelector',
    
    # Model Training
    'ModelTrainer',
    
    # Evaluation
    'ModelEvaluator',
    'calculate_optimal_threshold',
    'calculate_classification_metrics',
    'plot_roc_curves',
    
    # Explainability
    'SHAPExplainer',
    'create_shap_report',
    
    # Utils
    'save_model',
    'load_model',
    'save_results',
    'create_train_test_split',
    'DataScaler',
    'get_config',
]