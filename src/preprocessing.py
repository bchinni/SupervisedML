"""
Data preprocessing utilities for supervised ML pipeline.
Handles data cleaning, missingness analysis, and basic transformations.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def classify_columns(df: pd.DataFrame, threshold: int) -> Tuple[List[str], List[str]]:
    """
    Classify columns as numerical or categorical based on unique value count.
    
    Args:
        df: Input DataFrame
        threshold: Minimum unique values for numerical classification
        
    Returns:
        Tuple of (numerical_cols, categorical_cols)
    """
    numerical_cols = []
    categorical_cols = []

    for column in df.columns:
        unique_count = df[column].nunique()
        if unique_count >= threshold:
            numerical_cols.append(column)
        else:
            categorical_cols.append(column)

    return numerical_cols, categorical_cols


def replace_2_with_0_if_binary(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    For binary columns containing only 1, 2, and NaN, replace 2 with 0.
    
    Args:
        df: Input DataFrame
        columns: Specific columns to check (None = all numeric columns)
        
    Returns:
        Modified DataFrame
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        vals = set(df[col].dropna().unique())
        if vals.issubset({1, 2}):
            df[col] = df[col].replace(2, 0)
    
    return df


def clean_dataframe(df: pd.DataFrame, outcome_col: str, 
                    outcome_mapping: Optional[dict] = None) -> pd.DataFrame:
    """
    Perform initial data cleaning steps.
    
    Args:
        df: Input DataFrame
        outcome_col: Name of outcome column
        outcome_mapping: Optional mapping for outcome values
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Standardize column names
    df.columns = df.columns.str.lower()
    
    # Replace various null representations
    null_values = ['.', 'NaT', '', ' ']
    for null_val in null_values:
        df = df.replace(null_val, np.nan)
    
    # Convert binary 2s to 0s
    df = replace_2_with_0_if_binary(df)
    
    # Map outcome if provided
    if outcome_mapping:
        df[outcome_col] = df[outcome_col].map(outcome_mapping)
    
    # Drop fully NaN columns and rows
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='all')
    
    # Drop rows with missing outcome
    df = df.dropna(subset=[outcome_col])
    
    # Convert to numeric where possible
    df = df.apply(pd.to_numeric, errors='ignore')
    
    return df


def analyze_missingness(df: pd.DataFrame, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Analyze and report missing data patterns.
    
    Args:
        df: Input DataFrame
        output_path: Optional path to save Excel file
        
    Returns:
        DataFrame with missingness statistics
    """
    prop_miss = pd.DataFrame(df.isnull().sum(), columns=['MissingRows'])
    prop_miss['Proportion'] = prop_miss['MissingRows'] / len(df)
    
    if output_path:
        prop_miss.to_excel(output_path)
    
    return prop_miss


def drop_high_missing_vars(df: pd.DataFrame, threshold: float = 0.4) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop variables with missingness above threshold.
    
    Args:
        df: Input DataFrame
        threshold: Maximum allowed proportion of missing values
        
    Returns:
        Tuple of (cleaned DataFrame, list of dropped variables)
    """
    prop_miss = analyze_missingness(df)
    vars_to_drop = prop_miss[prop_miss['Proportion'] > threshold].index.tolist()
    
    df_cleaned = df.drop(vars_to_drop, axis=1)
    
    return df_cleaned, vars_to_drop


def split_and_prepare_data(df: pd.DataFrame, outcome_col: str, 
                           id_col: str = 'id') -> Tuple[pd.DataFrame, pd.Series, float]:
    """
    Prepare features and target, calculate prevalence.
    
    Args:
        df: Input DataFrame
        outcome_col: Name of outcome column
        id_col: Name of ID column
        
    Returns:
        Tuple of (features DataFrame, target Series, outcome prevalence)
    """
    X = df.copy()
    y = X[outcome_col]
    prevalence = y.mean()
    
    # Set index
    if id_col in X.columns:
        X.set_index([id_col, outcome_col], inplace=True)
    else:
        X.set_index([outcome_col], inplace=True)
    
    return X, y, prevalence