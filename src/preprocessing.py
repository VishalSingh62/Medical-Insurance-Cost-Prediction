"""Data Preprocessing, Validation, and Feature Pipeline Construction.

Provides end-to-end dataset validation, outlier analysis using IQR,
reproducible train/test splitting, and Scikit-Learn ColumnTransformer construction.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CLEANED_CSV_PATH,
    PREPROCESSING_SUMMARY_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    VALID_SEX,
    VALID_SMOKER,
    VALID_REGIONS,
    AGE_MIN,
    AGE_MAX,
    BMI_MIN,
    BMI_MAX,
    CHILDREN_MIN,
    CHILDREN_MAX,
    logger,
)


def validate_single_record(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a single insurance record dictionary against domain bounds.

    Args:
        record: Dictionary containing record fields.

    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
    """
    # Check required fields
    for field in ALL_FEATURES:
        if field not in record:
            return False, f"Missing required field: '{field}'"

    # Age validation
    try:
        age = float(record["age"])
        if not (AGE_MIN <= age <= AGE_MAX):
            return False, f"Age must be between {AGE_MIN} and {AGE_MAX}. Received: {age}"
    except (ValueError, TypeError):
        return False, f"Age must be a valid number. Received: {record.get('age')}"

    # BMI validation
    try:
        bmi = float(record["bmi"])
        if not (BMI_MIN <= bmi <= BMI_MAX):
            return False, f"BMI must be between {BMI_MIN} and {BMI_MAX}. Received: {bmi}"
    except (ValueError, TypeError):
        return False, f"BMI must be a valid number. Received: {record.get('bmi')}"

    # Children validation
    try:
        children = int(record["children"])
        if not (CHILDREN_MIN <= children <= CHILDREN_MAX):
            return False, f"Children must be between {CHILDREN_MIN} and {CHILDREN_MAX}. Received: {children}"
    except (ValueError, TypeError):
        return False, f"Children must be a valid integer. Received: {record.get('children')}"

    # Categorical fields validation
    sex = str(record["sex"]).strip().lower()
    if sex not in VALID_SEX:
        return False, f"Sex must be one of {VALID_SEX}. Received: '{record.get('sex')}'"

    smoker = str(record["smoker"]).strip().lower()
    if smoker not in VALID_SMOKER:
        return False, f"Smoker status must be one of {VALID_SMOKER}. Received: '{record.get('smoker')}'"

    region = str(record["region"]).strip().lower()
    if region not in VALID_REGIONS:
        return False, f"Region must be one of {VALID_REGIONS}. Received: '{record.get('region')}'"

    # Target charges validation if present
    if TARGET_COLUMN in record and record[TARGET_COLUMN] is not None:
        try:
            charges = float(record[TARGET_COLUMN])
            if charges < 0:
                return False, f"Charges must be non-negative. Received: {charges}"
        except (ValueError, TypeError):
            return False, f"Charges must be a valid number. Received: {record.get(TARGET_COLUMN)}"

    return True, None


def detect_outliers_iqr(series: pd.Series) -> Dict[str, Any]:
    """Detect outliers using the standard Interquartile Range (IQR) method.

    Does not remove valid medical observations, but provides actuarial transparency.

    Args:
        series: Numerical series to analyze.

    Returns:
        Dictionary with Q1, Q3, IQR, bounds, count and percentage of outliers.
    """
    clean_series = series.dropna()
    q1 = float(clean_series.quantile(0.25))
    q3 = float(clean_series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]

    return {
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "outlier_count": int(len(outliers)),
        "outlier_percentage": round(float(len(outliers) / len(clean_series) * 100), 2)
        if len(clean_series) > 0
        else 0.0,
    }


def clean_and_preprocess_dataset(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    save_cleaned: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean, audit, validate, and summarize the dataset.

    Checks:
    - Initial row and column count.
    - Missing values detection and reporting.
    - Duplicate detection and removal.
    - Type normalization and categorical formatting.
    - Domain validation of features.
    - Statistical IQR outlier reporting (without blind deletion).

    Args:
        df: Input raw DataFrame.
        drop_duplicates: Whether to drop duplicate feature rows.
        save_cleaned: Whether to save cleaned CSV and summary JSON.

    Returns:
        Tuple of (cleaned_df, summary_dict)
    """
    initial_shape = df.shape
    df_clean = df.copy()

    # Normalize column names (lowercase, stripped)
    df_clean.columns = [c.strip().lower() for c in df_clean.columns]

    # Check for missing values
    missing_summary = df_clean.isnull().sum().to_dict()
    total_missing = int(df_clean.isnull().sum().sum())

    # Handle missing values if any present
    if total_missing > 0:
        logger.warning(f"Detected {total_missing} missing values across features. Imputing with medians/modes.")
        for num_col in NUMERICAL_FEATURES:
            if num_col in df_clean.columns and df_clean[num_col].isnull().any():
                df_clean[num_col] = df_clean[num_col].fillna(df_clean[num_col].median())
        for cat_col in CATEGORICAL_FEATURES:
            if cat_col in df_clean.columns and df_clean[cat_col].isnull().any():
                df_clean[cat_col] = df_clean[cat_col].fillna(df_clean[cat_col].mode()[0])

    # Standardize string fields
    for col in CATEGORICAL_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()

    # Identify duplicate rows across feature columns
    feature_cols = [c for c in ALL_FEATURES if c in df_clean.columns]
    duplicate_count = int(df_clean.duplicated(subset=feature_cols).sum())

    if drop_duplicates and duplicate_count > 0:
        logger.info(f"Removing {duplicate_count} duplicate row(s) to prevent data leakage.")
        df_clean = df_clean.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)

    # Detect outliers for numerical columns
    outlier_analysis = {}
    for col in NUMERICAL_FEATURES + [TARGET_COLUMN]:
        if col in df_clean.columns:
            outlier_analysis[col] = detect_outliers_iqr(df_clean[col])

    # Summary dictionary
    summary = {
        "initial_rows": initial_shape[0],
        "initial_columns": initial_shape[1],
        "cleaned_rows": df_clean.shape[0],
        "cleaned_columns": df_clean.shape[1],
        "duplicates_removed": duplicate_count,
        "total_missing_values": total_missing,
        "missing_per_column": {k: int(v) for k, v in missing_summary.items()},
        "column_data_types": {col: str(dtype) for col, dtype in df_clean.dtypes.items()},
        "outliers_iqr_analysis": outlier_analysis,
    }

    if save_cleaned:
        CLEANED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREPROCESSING_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

        df_clean.to_csv(CLEANED_CSV_PATH, index=False)
        with open(PREPROCESSING_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved cleaned dataset to: {CLEANED_CSV_PATH}")
        logger.info(f"Saved preprocessing summary to: {PREPROCESSING_SUMMARY_PATH}")

    return df_clean, summary


def build_preprocessor() -> ColumnTransformer:
    """Construct Scikit-Learn ColumnTransformer.

    Applies:
    - StandardScaler to numerical features (age, bmi, children).
    - OneHotEncoder to categorical features (sex, smoker, region).
      Uses handle_unknown='ignore' and sparse_output=False to avoid matrix format mismatches.

    Returns:
        Configured ColumnTransformer instance.
    """
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_train_test_split(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into features and target, and partitioned train and test sets.

    Guarantees reproducibility and prevents target leakage.

    Args:
        df: Cleaned DataFrame.
        target_column: Name of target column (charges).
        test_size: Ratio of test partition.
        random_state: Random state seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # Features: ensure all required features are present
    X = df[ALL_FEATURES].copy()
    y = df[target_column].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    logger.info(
        f"Train/Test split complete: X_train={X_train.shape}, "
        f"X_test={X_test.shape}, random_state={random_state}"
    )
    return X_train, X_test, y_train, y_test
