"""
Model evaluation utilities including metrics, calibration, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc,
    precision_recall_fscore_support, f1_score, cohen_kappa_score,
    calibration_curve
)
from typing import Tuple, Dict, Any, Optional


def calculate_optimal_threshold(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """
    Calculate optimal classification threshold using Youden's J statistic.
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        
    Returns:
        Optimal threshold value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    youden_j = tpr - fpr
    best_threshold_index = np.argmax(youden_j)
    return thresholds[best_threshold_index]


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                     y_pred_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_prob: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    ppv, sens, _, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred_prob),
        'TN': tn, 'TP': tp, 'FN': fn, 'FP': fp,
        'PPV': ppv[1],  # Precision for positive class
        'Sensitivity': sens[1],  # Recall for positive class
        'Specificity': sens[0],  # Recall for negative class
        'NPV': ppv[0],  # Precision for negative class
        'FPR': fp / (fp + tn),
        'FNR': fn / (tp + fn),
        'FDR': fp / (tp + fp) if (tp + fp) > 0 else 0,
        'F1_score': f1_score(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
    }
    
    return {k: np.round(v, 2) for k, v in metrics.items()}


def calculate_discrimination_slope(y_true: np.ndarray, 
                                   y_pred_prob: np.ndarray) -> float:
    """
    Calculate discrimination slope (mean difference in predicted probabilities).
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        
    Returns:
        Discrimination slope value
    """
    avg_with_outcome = np.mean(y_pred_prob[y_true == 1])
    avg_without_outcome = np.mean(y_pred_prob[y_true == 0])
    return abs(avg_with_outcome - avg_without_outcome)


def calculate_calibration_slope(y_true: np.ndarray, 
                                y_pred_prob: np.ndarray, 
                                n_bins: int = 10) -> float:
    """
    Calculate calibration slope.
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        n_bins: Number of calibration bins
        
    Returns:
        Calibration slope value
    """
    observed_probs, predicted_probs = calibration_curve(
        y_true, y_pred_prob, n_bins=n_bins, strategy='quantile'
    )
    calibration_model = np.polyfit(predicted_probs, observed_probs, 1)
    return calibration_model[0]


def calculate_oe_ratio(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Tuple[float, pd.DataFrame]:
    """
    Calculate observed-to-expected ratio by deciles.
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        
    Returns:
        Tuple of (O/E ratio, decile statistics DataFrame)
    """
    probs_risk = pd.DataFrame({
        'PredictedProbs': y_pred_prob,
        'TrueLabels': y_true
    })
    probs_risk = probs_risk.sort_values(by='PredictedProbs', ascending=True)
    
    try:
        probs_risk['decile_rank'] = pd.qcut(
            probs_risk['PredictedProbs'], q=10, labels=False
        )
    except ValueError:
        probs_risk['decile_rank'] = pd.qcut(
            probs_risk['PredictedProbs'].rank(method='first'), q=10, labels=False
        )
    
    dec_stats = probs_risk.groupby('decile_rank').agg({
        'TrueLabels': 'sum',
        'PredictedProbs': 'sum'
    }).reset_index()
    
    dec_stats['abs_diff'] = abs(dec_stats['TrueLabels'] - dec_stats['PredictedProbs'])
    oe_ratio = np.round(dec_stats['TrueLabels'].sum() / dec_stats['PredictedProbs'].sum(), 3)
    
    return oe_ratio, dec_stats


def permutation_test_auc(y_true: np.ndarray, y_pred_prob: np.ndarray, 
                        n_permutations: int = 1000) -> float:
    """
    Perform permutation test for AUC significance.
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        n_permutations: Number of permutations
        
    Returns:
        P-value
    """
    n = len(y_true)
    k = round(n / 2)
    arr = np.array([0] * k + [1] * (n - k))
    
    observed_auc = roc_auc_score(y_true, y_pred_prob)
    
    permuted_aucs = []
    for _ in range(n_permutations):
        np.random.shuffle(arr)
        permuted_aucs.append(roc_auc_score(arr, y_pred_prob))
    
    p_value = len(np.where(np.array(permuted_aucs) >= observed_auc)[0]) / n_permutations
    return np.round(p_value, 3)


def plot_calibration_curve(y_true: np.ndarray, y_pred_prob: np.ndarray, 
                           model_name: str, save_path: Optional[str] = None) -> None:
    """
    Plot calibration curve.
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        model_name: Model name for title
        save_path: Optional path to save figure
    """
    prob_true, prob_predicted = calibration_curve(
        y_true, y_pred_prob, n_bins=10, strategy='quantile'
    )
    
    effective_min = min(np.min(prob_true), np.min(prob_predicted))
    effective_min = 0 if effective_min == 0 else effective_min - 0.01
    effective_max = max(np.max(prob_true), np.max(prob_predicted)) + 0.01
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle='--', label="Ideal")
    ax.plot(prob_predicted, prob_true, marker='.', label="Model calibration")
    ax.set_xlabel('Predicted Probabilities')
    ax.set_ylabel('True Probabilities')
    ax.set_title(f'Calibration Curve - {model_name}')
    ax.set_xlim([effective_min, effective_max])
    ax.set_ylim([effective_min, effective_max])
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()


def plot_roc_curves(models_dict: Dict[str, Any], X_test: pd.DataFrame, 
                   y_test: pd.Series, save_path: Optional[str] = None) -> None:
    """
    Plot ROC curves for multiple models.
    
    Args:
        models_dict: Dictionary of {name: model} pairs
        X_test: Test features
        y_test: Test target
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'purple', 'brown', 'orange', 'black', 'red']
    
    for (name, model), color in zip(models_dict.items(), colors):
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, 
                label=f'{name.upper()} (AUC = {np.round(roc_auc, 2)})')
    
    plt.plot([0, 1], [0, 1], 'y--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='best')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()


class ModelEvaluator:
    """Comprehensive model evaluation wrapper."""
    
    def __init__(self):
        self.results = None
        
    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series,
                      model_name: str, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluate a single model comprehensively.
        
        Args:
            model: Trained model
            X: Features
            y: Target
            model_name: Name for identification
            threshold: Classification threshold (if None, calculate optimal)
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred_prob = model.predict_proba(X)[:, 1]
        
        if threshold is None:
            threshold = calculate_optimal_threshold(y.values, y_pred_prob)
        
        y_pred = (y_pred_prob > threshold).astype(int)
        
        metrics = calculate_classification_metrics(y.values, y_pred, y_pred_prob)
        metrics['threshold'] = np.round(threshold, 4)
        metrics['discrimination_slope'] = calculate_discrimination_slope(y.values, y_pred_prob)
        metrics['calibration_slope'] = calculate_calibration_slope(y.values, y_pred_prob)
        metrics['oe_ratio'], _ = calculate_oe_ratio(y.values, y_pred_prob)
        metrics['auc_p_val'] = permutation_test_auc(y.values, y_pred_prob)
        metrics['model_name'] = model_name
        
        return metrics
    
    def evaluate_multiple_models(self, models_dict: Dict[str, Any], 
                                X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate multiple models on train and test sets.
        
        Args:
            models_dict: Dictionary of {name: model} pairs
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            DataFrame with all evaluation results
        """
        results = []
        
        for name, model in models_dict.items():
            print(f"\nEvaluating {name.upper()}...")
            
            # Calculate threshold on training data
            y_train_pred_prob = model.predict_proba(X_train)[:, 1]
            threshold = calculate_optimal_threshold(y_train.values, y_train_pred_prob)
            
            # Evaluate on training set
            train_metrics = self.evaluate_model(model, X_train, y_train, name, threshold)
            train_metrics['cohort'] = 'train'
            results.append(train_metrics)
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(model, X_test, y_test, name, threshold)
            test_metrics['cohort'] = 'test'
            results.append(test_metrics)
        
        self.results = pd.DataFrame(results)
        return self.results