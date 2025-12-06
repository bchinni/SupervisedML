#Supervised Machine Learning Complete Cycle

Machine learning pipeline for binary classification tasks with automated feature selection, hyperparameter optimization, and model explainability.

Automated Data Preprocessing: Handles missing values, data cleaning, and type conversions
Iterative Imputation: Uses decision tree-based imputation for missing data
Feature Selection: Boruta-SHAP algorithm for robust feature selection
Multiple Models: Trains 6 different classifiers (LR, SVM, RF, XGBoost, LightGBM, Neural Network)
Bayesian Optimization: Automated hyperparameter tuning using scikit-optimize
Evaluation: AUC, calibration, discrimination metrics, and permutation tests
Explainability: SHAP values for feature importance and instance-level explanations


#Edit Configuration in src folder in utils file to customize
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

#PipelineExecutionExample

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


#Output Files
After running the pipeline, following files were generated in corresponding folders within output folder.
#Models (output/models/)
imputer.pkl - Fitted imputer
scaler.pkl - Fitted feature scaler
model_lr.pkl, model_xgb.pkl, etc. - Trained models


#Results (output/results/)
evaluation_results.xlsx - Comprehensive metrics for all models
selected_features.xlsx - Features selected by Boruta
missingness_report.xlsx - Missing data analysis


#Plots (output/plots/)
roc_curves_test.png - ROC curves for all models


#SHAP (output/shap/)
shap_bar_xgboost.png - Feature importance bar plot
shap_beeswarm_xgboost.png - Feature effect beeswarm plot
feature_importance_xgboost.csv - Detailed feature importance
explainer_xgboost.pkl - Saved SHAP explainer


#Models Trained
Logistic Regression - Linear baseline model
Support Vector Machine (SVM) - Non-linear kernel methods
Random Forest - Ensemble of decision trees
XGBoost - Gradient boosting with regularization
LightGBM - Fast gradient boosting
Neural Network - Multi-layer perceptron


#All models use Bayesian hyperparameter optimization for best performance.
#Evaluation Metrics

Discrimination: AUC, discrimination slope
Calibration: Calibration slope, O/E ratio
Classification: Accuracy, sensitivity, specificity, PPV, NPV
Statistical: Permutation test p-value, Cohen's kappa
Visualization: ROC curves, calibration curves, risk distributions
