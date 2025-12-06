"""
Main pipeline script for supervised ML analysis.
Orchestrates the complete workflow from data loading to model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.preprocessing import (
    clean_dataframe, analyze_missingness, drop_high_missing_vars,
    split_and_prepare_data, classify_columns
)
from src.imputation import DataImputer
from src.feature_selection import BorutaFeatureSelector
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator, plot_roc_curves
from src.explainability import create_shap_report
from src.utils import (
    create_train_test_split, DataScaler, setup_output_directory,
    get_config, print_dataset_summary, save_model, save_results
)


def main():
    """Main pipeline execution."""
    
    # 1. Setup
    print("\n" + "="*80)
    print("SUPERVISED ML PIPELINE")
    print("="*80)
    
    config = get_config()
    output_dirs = setup_output_directory(config['output_dir'])
    
    # 2. Load Data
    print("\n[1/9] Loading data...")
    data_path = 'data/data.csv'
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    outcome_col = 'diagnosis'
    id_col = 'id'
    
    # 3. Data Preprocessing
    print("\n[2/9] Preprocessing data...")
    df_clean = clean_dataframe(
        df, 
        outcome_col=outcome_col,
        outcome_mapping={'M': 1, 'B': 0}
    )
    
    # Analyze and handle missingness
    missingness = analyze_missingness(
        df_clean, 
        output_path=f"{output_dirs['results']}/missingness_report.xlsx"
    )
    print(f"Variables with >40% missing: {(missingness['Proportion'] > 0.4).sum()}")
    
    df_clean, dropped_vars = drop_high_missing_vars(
        df_clean, 
        threshold=config['missing_threshold']
    )
    
    # 4. Imputation
    print("\n[3/9] Imputing missing values...")
    X, y, prevalence = split_and_prepare_data(df_clean, outcome_col, id_col)
    
    imputer = DataImputer(
        max_iter=config['max_impute_iter'],
        random_state=config['random_state']
    )
    X_imputed, imputation_done = imputer.fit_transform(X)
    
    if imputation_done:
        save_model(imputer, f"{output_dirs['models']}/imputer.pkl")
    
    # Reset index for splitting
    X_imputed = X_imputed.reset_index(drop=False)
    
    # 5. Train-Test Split
    print("\n[4/9] Creating train-test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(
        X_imputed, y,
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
    # Separate ID column
    y_train_with_id = X_train[[id_col, outcome_col]]
    y_test_with_id = X_test[[id_col, outcome_col]]
    
    X_train = X_train.drop([outcome_col], axis=1)
    X_test = X_test.drop([outcome_col], axis=1)
    
    X_train.set_index([id_col], inplace=True)
    X_test.set_index([id_col], inplace=True)
    y_train_with_id.set_index([id_col], inplace=True)
    y_test_with_id.set_index([id_col], inplace=True)
    
    print_dataset_summary(X_train, y_train_with_id[outcome_col], "Training Set")
    print_dataset_summary(X_test, y_test_with_id[outcome_col], "Test Set")
    
    # 6. Feature Selection
    print("\n[5/9] Performing feature selection...")
    feature_selector = BorutaFeatureSelector(
        percentile=config['boruta_percentile'],
        pvalue=config['boruta_pvalue'],
        random_state=config['random_state']
    )
    
    X_train_selected = feature_selector.fit_transform(
        X_train, 
        y_train_with_id[outcome_col]
    )
    X_test_selected = feature_selector.transform(X_test)
    
    selected_features = feature_selector.selected_features
    pd.DataFrame({'selected_features': selected_features}).to_excel(
        f"{output_dirs['results']}/selected_features.xlsx", 
        index=False
    )
    
    # 7. Feature Scaling
    print("\n[6/9] Scaling features...")
    numerical_cols, categorical_cols = classify_columns(X_train_selected, threshold=3)
    
    scaler = DataScaler(numerical_cols, categorical_cols)
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    save_model(scaler, f"{output_dirs['models']}/scaler.pkl")
    
    # 8. Model Training
    print("\n[7/9] Training models...")
    trainer = ModelTrainer(
        n_iter=config['n_bayes_iter'],
        cv=config['cv_folds'],
        random_state=config['random_state']
    )
    
    training_results = trainer.train_all_models(
        X_train_scaled, 
        y_train_with_id[outcome_col]
    )
    
    # Save all models
    models_dict = {}
    for name, (model, params, score) in training_results.items():
        models_dict[name] = model
        save_model(model, f"{output_dirs['models']}/model_{name}.pkl")
        print(f"\n{name.upper()} - Best CV Score: {score:.4f}")
    
    # 9. Model Evaluation
    print("\n[8/9] Evaluating models...")
    evaluator = ModelEvaluator()
    results_df = evaluator.evaluate_multiple_models(
        models_dict,
        X_train_scaled, y_train_with_id[outcome_col],
        X_test_scaled, y_test_with_id[outcome_col]
    )
    
    # Add metadata
    results_df['n_features'] = len(selected_features)
    results_df['data_prevalence'] = prevalence
    
    save_results(results_df, f"{output_dirs['results']}/evaluation_results.xlsx")
    
    # Plot ROC curves
    plot_roc_curves(
        models_dict, 
        X_test_scaled, 
        y_test_with_id[outcome_col],
        save_path=f"{output_dirs['plots']}/roc_curves_test.png"
    )
    
    # 10. Model Explainability
    print("\n[9/9] Generating SHAP explanations...")
    
    # Focus on XGBoost for detailed SHAP analysis
    if 'xgb' in models_dict:
        importance_df = create_shap_report(
            models_dict['xgb'],
            X_train_scaled,
            output_dirs['shap'],
            'xgboost'
        )
        print(f"\nTop 5 Features (by SHAP):")
        print(importance_df.head(5)[['feature', 'mean_abs_shap']])
    
    # Final Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dirs['base']}/")
    print(f"- Models: {output_dirs['models']}/")
    print(f"- Results: {output_dirs['results']}/")
    print(f"- Plots: {output_dirs['plots']}/")
    print(f"- SHAP: {output_dirs['shap']}/")
    
    # Print best model on test set
    test_results = results_df[results_df['cohort'] == 'test']
    best_model = test_results.loc[test_results['AUC'].idxmax(), 'model_name']
    best_auc = test_results['AUC'].max()
    print(f"\nBest Test AUC: {best_auc:.3f} ({best_model.upper()})")
    
    return results_df, models_dict


if __name__ == "__main__":
    results, models = main()
    
    # Keep plots open
    plt.show()