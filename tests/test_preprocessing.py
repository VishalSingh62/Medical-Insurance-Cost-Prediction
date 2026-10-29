"""Unit tests for data preprocessing, domain validation, and feature splitting."""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing import (
    validate_single_record,
    detect_outliers_iqr,
    clean_and_preprocess_dataset,
    prepare_train_test_split,
    build_preprocessor,
)


def test_validate_single_record_valid():
    """Verify that a valid medical profile passes domain validation."""
    valid_doc = {
        "age": 28,
        "sex": "female",
        "bmi": 24.5,
        "children": 1,
        "smoker": "no",
        "region": "northwest",
        "charges": 3500.0,
    }
    is_valid, err = validate_single_record(valid_doc)
    assert is_valid is True
    assert err is None


def test_validate_single_record_invalid_age():
    """Verify that impossible ages (<18 or >120) fail validation."""
    bad_doc_young = {
        "age": 12,  # Too young
        "sex": "female",
        "bmi": 22.0,
        "children": 0,
        "smoker": "no",
        "region": "southwest",
    }
    is_valid, err = validate_single_record(bad_doc_young)
    assert is_valid is False
    assert "Age must be between" in err

    bad_doc_old = bad_doc_young.copy()
    bad_doc_old["age"] = 150
    is_valid, err = validate_single_record(bad_doc_old)
    assert is_valid is False


def test_validate_single_record_invalid_categorical():
    """Verify that unrecognized categories fail validation."""
    bad_smoker = {
        "age": 30,
        "sex": "male",
        "bmi": 25.0,
        "children": 0,
        "smoker": "sometimes",  # Invalid
        "region": "southwest",
    }
    is_valid, err = validate_single_record(bad_smoker)
    assert is_valid is False
    assert "Smoker status must be one of" in err

    bad_region = bad_smoker.copy()
    bad_region["smoker"] = "yes"
    bad_region["region"] = "atlantis"  # Invalid
    is_valid, err = validate_single_record(bad_region)
    assert is_valid is False
    assert "Region must be one of" in err


def test_detect_outliers_iqr():
    """Verify IQR calculation and bounds detection."""
    # Synthetic series with known extreme values
    s = pd.Series([10, 11, 12, 13, 14, 15, 100])
    res = detect_outliers_iqr(s)
    assert res["outlier_count"] == 1
    assert res["upper_bound"] < 100


def test_clean_and_preprocess_dataset_handles_duplicates():
    """Verify that duplicate rows are detected and removed."""
    sample_df = pd.DataFrame(
        [
            {"age": 25, "sex": "male", "bmi": 24.0, "children": 0, "smoker": "no", "region": "southwest", "charges": 3000.0},
            {"age": 25, "sex": "male", "bmi": 24.0, "children": 0, "smoker": "no", "region": "southwest", "charges": 3000.0},  # Duplicate
            {"age": 45, "sex": "female", "bmi": 32.0, "children": 2, "smoker": "yes", "region": "northeast", "charges": 25000.0},
        ]
    )
    cleaned, summary = clean_and_preprocess_dataset(sample_df, drop_duplicates=True, save_cleaned=False)
    assert len(cleaned) == 2
    assert summary["duplicates_removed"] == 1


def test_prepare_train_test_split_reproducibility():
    """Verify that train/test split is reproducible and partitions correctly."""
    data = []
    for i in range(100):
        data.append(
            {
                "age": 20 + (i % 50),
                "sex": "male" if i % 2 == 0 else "female",
                "bmi": 20.0 + (i % 20),
                "children": i % 4,
                "smoker": "yes" if i % 5 == 0 else "no",
                "region": ["southwest", "southeast", "northwest", "northeast"][i % 4],
                "charges": 2000.0 + (i * 100),
            }
        )
    df = pd.DataFrame(data)

    X_train_1, X_test_1, y_train_1, y_test_1 = prepare_train_test_split(df, test_size=0.2, random_state=42)
    X_train_2, X_test_2, y_train_2, y_test_2 = prepare_train_test_split(df, test_size=0.2, random_state=42)

    assert len(X_train_1) == 80
    assert len(X_test_1) == 20
    # Reproducibility check
    pd.testing.assert_frame_equal(X_train_1, X_train_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)


def test_build_preprocessor_transformation():
    """Verify Scikit-Learn ColumnTransformer transformation shapes."""
    preprocessor = build_preprocessor()
    sample_data = pd.DataFrame(
        [
            {"age": 30, "sex": "male", "bmi": 27.0, "children": 1, "smoker": "no", "region": "southwest"},
            {"age": 50, "sex": "female", "bmi": 34.0, "children": 0, "smoker": "yes", "region": "northeast"},
            {"age": 40, "sex": "male", "bmi": 30.0, "children": 2, "smoker": "no", "region": "northwest"},
            {"age": 22, "sex": "female", "bmi": 22.0, "children": 0, "smoker": "yes", "region": "southeast"},
        ]
    )
    transformed = preprocessor.fit_transform(sample_data)
    # 3 numerical features + 2 sex + 2 smoker + 4 region = 11 encoded features
    assert transformed.shape[0] == 4
    assert transformed.shape[1] == 11

