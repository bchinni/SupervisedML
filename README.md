# Supervised Machine Learning Complete Cycle

A comprehensive machine learning pipeline for binary classification tasks with automated feature selection, hyperparameter optimization, and model explainability.

## Features

- **Automated Data Preprocessing**: Handles missing values, data cleaning, and type conversions
- **Iterative Imputation**: Uses decision tree-based imputation for missing data
- **Feature Selection**: Boruta-SHAP algorithm for robust feature selection
- **Multiple Models**: Trains 6 different classifiers (LR, SVM, RF, XGBoost, LightGBM, Neural Network)
- **Bayesian Optimization**: Automated hyperparameter tuning using scikit-optimize
- **Comprehensive Evaluation**: AUC, calibration, discrimination metrics, and permutation tests
- **Model Explainability**: SHAP values for feature importance and instance-level explanations

## Requirements

```bash
pip install -r requirements.txt
```

## Configuration

Edit the configuration in `src/utils.py` to customize pipeline parameters:

```python
config = {
    'random_state': 42,           # Random seed
    'test_size': 0.2,             # Test set proportion
    'cv_folds': 5,                # Cross-validation folds
    'n_bayes_iter': 50,           # Bayesian optimization iterations
    'missing_threshold': 0.4,     # Drop vars with >40% missing
    'boruta_percentile': 90,      # Boruta feature selection threshold
    'boruta_pvalue': 0.05,        # Boruta p-value threshold
    'max_impute_iter': 100,       # Maximum imputation iterations
    'n_permutations': 1000,       # Permutation test iterations
    'output_dir': 'output'        # Output directory
}
```

## Quick Start

### Option 1: Run Complete Pipeline

```bash
python main.py
```

### Option 2: Use Individual Components

```python
from src.preprocessing import clean_dataframe, drop_high_missing_vars
from src.imputation import DataImputer
from src.feature_selection import BorutaFeatureSelector
from src.model_training import ModelTrainer

# Load and clean data
df = pd.read_csv('data/data.csv')
df_clean = clean_dataframe(df, outcome_col='diagnosis', 
                           outcome_mapping={'M': 1, 'B': 0})

# Impute missing values
imputer = DataImputer()
X_imputed, _ = imputer.fit_transform(X)

# Feature selection
selector = BorutaFeatureSelector(percentile=90)
X_selected = selector.fit_transform(X_train, y_train)

# Train models
trainer = ModelTrainer(n_iter=50, cv=5)
results = trainer.train_all_models(X_train, y_train)
```

## Output Files

After running the pipeline, the following files are generated:

### Models (`output/models/`)
- `imputer.pkl` - Fitted imputer
- `scaler.pkl` - Fitted feature scaler
- `model_lr.pkl`, `model_xgb.pkl`, etc. - Trained models

### Results (`output/results/`)
- `evaluation_results.xlsx` - Comprehensive metrics for all models
- `selected_features.xlsx` - Features selected by Boruta
- `missingness_report.xlsx` - Missing data analysis

### Plots (`output/plots/`)
- `roc_curves_test.png` - ROC curves for all models

### SHAP Analysis (`output/shap/`)
- `shap_bar_xgboost.png` - Feature importance bar plot
- `shap_beeswarm_xgboost.png` - Feature effect beeswarm plot
- `feature_importance_xgboost.csv` - Detailed feature importance
- `explainer_xgboost.pkl` - Saved SHAP explainer

## 🤖 Models Trained

All models use **Bayesian hyperparameter optimization** for best performance:

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear baseline model |
| **Support Vector Machine (SVM)** | Non-linear kernel methods |
| **Random Forest** | Ensemble of decision trees |
| **XGBoost** | Gradient boosting with regularization |
| **LightGBM** | Fast gradient boosting |
| **Neural Network** | Multi-layer perceptron |

## Evaluation Metrics

### Discrimination
- AUC (Area Under the ROC Curve)
- Discrimination slope

### Calibration
- Calibration slope
- Observed-to-Expected (O/E) ratio

### Classification Performance
- Accuracy
- Sensitivity (Recall)
- Specificity
- Positive Predictive Value (PPV/Precision)
- Negative Predictive Value (NPV)

### Statistical Tests
- Permutation test p-value
- Cohen's kappa

### Visualizations
- ROC curves
- Calibration curves
- Risk distributions

## 📁 Project Structure

```
SupervisedML/
├── data/                       # Place your data.csv here
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Data cleaning, missingness analysis
│   ├── imputation.py           # IterativeImputer utilities
│   ├── feature_selection.py    # Boruta-SHAP wrapper
│   ├── model_training.py       # Model training with Bayesian optimization
│   ├── evaluation.py           # Metrics, ROC, calibration curves
│   ├── explainability.py       # SHAP analysis
│   └── utils.py                # General helpers (I/O, config, scaling)
├── output/                     # Generated outputs
│   ├── models/                 # Saved trained models
│   ├── results/                # CSV/Excel results
│   ├── plots/                  # Visualization outputs
│   └── shap/                   # SHAP analysis outputs
├── main.py                     # Main pipeline script
├── requirements.txt
└── README.md
```

## Usage Examples

### Model Evaluation

```python
from src.evaluation import ModelEvaluator, plot_roc_curves

evaluator = ModelEvaluator()
results_df = evaluator.evaluate_multiple_models(
    models_dict, X_train, y_train, X_test, y_test
)

plot_roc_curves(models_dict, X_test, y_test, 
                save_path='roc_curves.png')
```

### SHAP Explainability

```python
from src.explainability import SHAPExplainer

explainer = SHAPExplainer(model, X_train)
explainer.calculate_shap_values(X_test)
explainer.plot_bar(max_display=15)
explainer.plot_beeswarm(max_display=15)

# Explain single instance
instance_explanation = explainer.explain_instance(
    X_test.iloc[[0]], n_top_features=5
)
```

## Tips

1. **Data Format**: Ensure your `data.csv` includes:
   - An `id` column for sample identifiers
   - An outcome column (default: `diagnosis`)
   - Feature columns

2. **Customization**: Modify the configuration in `src/utils.py` before running the pipeline

3. **Model Selection**: Review `evaluation_results.xlsx` to compare model performance

4. **Feature Importance**: Check SHAP outputs to understand feature contributions

**Note**: All models are trained using Bayesian optimization to ensure optimal hyperparameter selection for your specific dataset.
