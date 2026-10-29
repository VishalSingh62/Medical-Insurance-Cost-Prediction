"""Unit tests for regression evaluation metrics and model comparison logic."""

import pytest
import numpy as np
import pandas as pd

from src.evaluation import compute_regression_metrics, compare_models
from src.prediction import predict_insurance_charge


def test_compute_regression_metrics_perfect_fit():
    """Verify metrics for perfect predictions (R²=1.0, MAE=0.0, RMSE=0.0)."""
    y_true = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    y_pred = np.array([1000.0, 2000.0, 3000.0, 4000.0])

    metrics = compute_regression_metrics(y_true, y_pred, model_name="Perfect")
    assert metrics["r2_score"] == 1.0
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mape_percent"] == 0.0


def test_compute_regression_metrics_known_residuals():
    """Verify MAE and RMSE calculation with known residual offsets."""
    y_true = np.array([100.0, 200.0])
    # Predicted is off by +10 each
    y_pred = np.array([110.0, 210.0])

    metrics = compute_regression_metrics(y_true, y_pred, model_name="Offset")
    assert metrics["mae"] == 10.0
    assert metrics["rmse"] == 10.0


def test_compute_regression_metrics_mismatched_lengths():
    """Verify that mismatched array lengths raise ValueError."""
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([10.0])
    with pytest.raises(ValueError):
        compute_regression_metrics(y_true, y_pred)


def test_compute_regression_metrics_empty_array():
    """Verify that empty inputs raise ValueError."""
    with pytest.raises(ValueError):
        compute_regression_metrics(np.array([]), np.array([]))


def test_compare_models_selection():
    """Verify that compare_models selects the model with higher test R²."""
    mock_results = {
        "model_a": {
            "model_name": "Model Alpha",
            "test_r2": 0.72,
            "test_mae": 4500.0,
            "test_rmse": 6200.0,
            "test_mape_percent": 35.0,
        },
        "model_b": {
            "model_name": "Model Beta",
            "test_r2": 0.86,  # Higher R2
            "test_mae": 2700.0,
            "test_rmse": 4500.0,
            "test_mape_percent": 22.0,
        },
    }

    df_comp, best_key = compare_models(mock_results, save_outputs=False)
    assert best_key == "model_b"
    assert df_comp.iloc[0]["model_key"] == "model_b"
    assert df_comp.iloc[0]["test_r2"] == 0.86


def test_predict_insurance_charge_input_validation():
    """Verify that predict_insurance_charge catches invalid inputs before calling model."""
    bad_profile = {
        "age": 10,  # Below minimum 18
        "sex": "male",
        "bmi": 25.0,
        "children": 0,
        "smoker": "no",
        "region": "southwest",
    }
    with pytest.raises(ValueError, match="Age must be between"):
        predict_insurance_charge(bad_profile)
