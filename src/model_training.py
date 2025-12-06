"""
Model training utilities with Bayesian hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from typing import Dict, Any, Tuple


class ModelTrainer:
    """Orchestrates training of multiple models with Bayesian optimization."""
    
    def __init__(self, n_iter: int = 50, cv: int = 5, 
                 n_jobs: int = -1, random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            n_iter: Number of Bayesian optimization iterations
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        self.n_iter = n_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.best_scores = {}
        
    def get_param_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Get hyperparameter search spaces for all models."""
        return {
            'lr': {
                'C': (0.001, 10.0, 'log-uniform'),
                'fit_intercept': [True, False],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'warm_start': [False, True],
                'class_weight': [None, 'balanced'],
            },
            'svc': {
                'C': (0.001, 10.0, 'log-uniform'),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': (1, 8),
                'gamma': (0.001, 1.0, 'log-uniform'),
                'class_weight': [None, 'balanced'],
            },
            'rf': {
                'n_estimators': (100, 500),
                'max_depth': (3, 8),
                'min_samples_split': (20, 30),
                'min_samples_leaf': (5, 10),
                'max_features': (0.1, 1.0, 'uniform'),
                'class_weight': [None, 'balanced'],
            },
            'xgb': {
                'n_estimators': (100, 500),
                'max_depth': (3, 8),
                'learning_rate': (0.01, 1.0, 'log-uniform'),
                'subsample': (0.5, 1.0, 'uniform'),
                'colsample_bytree': (0.5, 1.0, 'uniform'),
                'gamma': (0.01, 1.0, 'log-uniform'),
                'alpha': (0.01, 2.0, 'log-uniform'),
                'lambda': (0.01, 2.0, 'log-uniform'),
                'min_child_weight': (3, 10),
                'objective': ['binary:logistic'],
                'eval_metric': ['auc'],
            },
            'lgbm': {
                'n_estimators': (100, 500),
                'max_depth': (3, 8),
                'learning_rate': (0.01, 1.0, 'log-uniform'),
                'subsample': (0.5, 1.0, 'uniform'),
                'colsample_bytree': (0.5, 1.0, 'uniform'),
                'min_child_samples': (5, 10),
                'reg_alpha': (0, 10, 'uniform'),
                'reg_lambda': (0, 10, 'uniform'),
                'objective': ['binary'],
            },
            'nn': {
                'hidden_layer_sizes': Integer(1, 25),
                'activation': ['relu', 'tanh'],
                'alpha': Real(0.001, 1),
                'solver': Categorical(['adam', 'sgd', 'lbfgs']),
                'learning_rate_init': Real(0.01, 0.3),
                'batch_size': Categorical([16, 32, 64, 128]),
            }
        }
    
    def train_logistic_regression(self, X_train: pd.DataFrame, 
                                  y_train: pd.Series) -> Tuple[Any, dict, float]:
        """Train Logistic Regression with Bayesian optimization."""
        param_space = self.get_param_spaces()['lr']
        model = LogisticRegression(random_state=self.random_state)
        
        optimizer = BayesSearchCV(
            model, param_space, n_iter=self.n_iter, cv=self.cv,
            n_jobs=self.n_jobs, verbose=1, random_state=self.random_state
        )
        
        optimizer.fit(X_train, y_train)
        
        return optimizer.best_estimator_, optimizer.best_params_, optimizer.best_score_
    
    def train_svm(self, X_train: pd.DataFrame, 
                  y_train: pd.Series) -> Tuple[Any, dict, float]:
        """Train SVM with Bayesian optimization."""
        param_space = self.get_param_spaces()['svc']
        model = SVC(probability=True, random_state=self.random_state)
        
        optimizer = BayesSearchCV(
            model, param_space, n_iter=self.n_iter, cv=self.cv,
            n_jobs=self.n_jobs, verbose=1, random_state=self.random_state
        )
        
        optimizer.fit(X_train, y_train)
        
        return optimizer.best_estimator_, optimizer.best_params_, optimizer.best_score_
    
    def train_random_forest(self, X_train: pd.DataFrame, 
                           y_train: pd.Series) -> Tuple[Any, dict, float]:
        """Train Random Forest with Bayesian optimization."""
        param_space = self.get_param_spaces()['rf']
        model = RandomForestClassifier(random_state=self.random_state)
        
        optimizer = BayesSearchCV(
            model, param_space, n_iter=self.n_iter, cv=self.cv,
            n_jobs=self.n_jobs, verbose=1, random_state=self.random_state
        )
        
        optimizer.fit(X_train, y_train)
        
        return optimizer.best_estimator_, optimizer.best_params_, optimizer.best_score_
    
    def train_xgboost(self, X_train: pd.DataFrame, 
                     y_train: pd.Series) -> Tuple[Any, dict, float]:
        """Train XGBoost with Bayesian optimization."""
        param_space = self.get_param_spaces()['xgb']
        model = XGBClassifier(random_state=self.random_state)
        
        optimizer = BayesSearchCV(
            model, param_space, n_iter=self.n_iter, cv=self.cv,
            n_jobs=self.n_jobs, verbose=1, random_state=self.random_state
        )
        
        optimizer.fit(X_train, y_train)
        
        return optimizer.best_estimator_, optimizer.best_params_, optimizer.best_score_
    
    def train_lightgbm(self, X_train: pd.DataFrame, 
                      y_train: pd.Series) -> Tuple[Any, dict, float]:
        """Train LightGBM with Bayesian optimization."""
        param_space = self.get_param_spaces()['lgbm']
        model = LGBMClassifier(random_state=self.random_state)
        
        optimizer = BayesSearchCV(
            model, param_space, n_iter=self.n_iter, cv=self.cv,
            n_jobs=self.n_jobs, verbose=1, random_state=self.random_state
        )
        
        optimizer.fit(X_train, y_train)
        
        return optimizer.best_estimator_, optimizer.best_params_, optimizer.best_score_
    
    def train_neural_network(self, X_train: pd.DataFrame, 
                            y_train: pd.Series) -> Tuple[Any, dict, float]:
        """Train Neural Network with Bayesian optimization."""
        param_space = self.get_param_spaces()['nn']
        model = MLPClassifier(max_iter=100, activation='logistic', 
                             random_state=self.random_state)
        
        optimizer = BayesSearchCV(
            model, param_space, n_iter=self.n_iter, cv=self.cv,
            scoring='roc_auc', n_jobs=self.n_jobs, verbose=1,
            random_state=self.random_state
        )
        
        optimizer.fit(X_train, y_train)
        
        return optimizer.best_estimator_, optimizer.best_params_, optimizer.best_score_
    
    def train_all_models(self, X_train: pd.DataFrame, 
                        y_train: pd.Series) -> Dict[str, Tuple[Any, dict, float]]:
        """
        Train all models and return results.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary with model results
        """
        results = {}
        
        print("\n" + "="*60)
        print("Training Logistic Regression")
        print("="*60)
        results['lr'] = self.train_logistic_regression(X_train, y_train)
        
        print("\n" + "="*60)
        print("Training SVM")
        print("="*60)
        results['svc'] = self.train_svm(X_train, y_train)
        
        print("\n" + "="*60)
        print("Training Random Forest")
        print("="*60)
        results['rf'] = self.train_random_forest(X_train, y_train)
        
        print("\n" + "="*60)
        print("Training XGBoost")
        print("="*60)
        results['xgb'] = self.train_xgboost(X_train, y_train)
        
        print("\n" + "="*60)
        print("Training LightGBM")
        print("="*60)
        results['lgbm'] = self.train_lightgbm(X_train, y_train)
        
        print("\n" + "="*60)
        print("Training Neural Network")
        print("="*60)
        results['nn'] = self.train_neural_network(X_train, y_train)
        
        return results