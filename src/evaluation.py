"""Model Evaluation and Regression Metrics Module.

Calculates continuous regression metrics (R², MAE, RMSE), produces side-by-side
model comparison summaries, and exports metrics to JSON and CSV formats.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.config import (
    METRICS_JSON_PATH,
    METRICS_CSV_PATH,
    BEST_MODEL_METADATA_PATH,
    logger,
)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Compute R², MAE, and RMSE regression metrics.

    ACADEMIC NOTE ON CLASSIFICATION METRICS:
    Accuracy, precision, recall, and F1-score evaluate categorical classifications
    against a discrete confusion matrix (True Positives, False Positives, etc.).
    Medical insurance charges are continuous numerical values (dollars), where
    exact floating-point matches are impossible. Thus, continuous loss functions
    (MAE, RMSE, and variance explained R²) are the only mathematically valid metrics.

    Args:
        y_true: Ground truth target charges.
        y_pred: Model predicted target charges.
        model_name: Descriptive name of the algorithm.

    Returns:
        Dictionary of computed metric values.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    if len(y_t) == 0 or len(y_p) == 0:
        raise ValueError("Target arrays must not be empty.")
    if len(y_t) != len(y_p):
        raise ValueError(f"Array length mismatch: {len(y_t)} vs {len(y_p)}")

    r2 = float(r2_score(y_t, y_p))
    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))

    # Mean Absolute Percentage Error (MAPE)
    non_zero = y_t != 0
    mape = float(np.mean(np.abs((y_t[non_zero] - y_p[non_zero]) / y_t[non_zero])) * 100)

    metrics = {
        "model_name": model_name,
        "sample_count": len(y_t),
        "r2_score": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_percent": round(mape, 2),
    }
    return metrics


def compare_models(
    model_results: Dict[str, Dict[str, Any]],
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Compare multiple model evaluation outputs and determine the best-performing algorithm.

    Selection Rule:
    The primary criterion is highest Test R² Score.
    In case of near ties, lowest RMSE and MAE break the tie.

    Args:
        model_results: Dict keyed by model identifier containing evaluation metrics.
        save_outputs: Whether to write outputs to JSON and CSV.

    Returns:
        Tuple of (comparison_dataframe: pd.DataFrame, best_model_key: str)
    """
    records: List[Dict[str, Any]] = []
    for key, metrics in model_results.items():
        rec = {
            "model_key": key,
            "algorithm": metrics.get("model_name", key),
            "test_r2": metrics.get("test_r2", metrics.get("r2_score")),
            "train_r2": metrics.get("train_r2", None),
            "test_mae": metrics.get("test_mae", metrics.get("mae")),
            "test_rmse": metrics.get("test_rmse", metrics.get("rmse")),
            "test_mape_pct": metrics.get("test_mape_percent", metrics.get("mape_percent")),
        }
        records.append(rec)

    df_comp = pd.DataFrame(records)
    # Sort descending by test_r2, ascending by test_rmse
    df_comp = df_comp.sort_values(by=["test_r2", "test_rmse"], ascending=[False, True]).reset_index(drop=True)

    best_model_key = str(df_comp.iloc[0]["model_key"])
    best_algorithm = str(df_comp.iloc[0]["algorithm"])
    best_r2 = float(df_comp.iloc[0]["test_r2"])
    best_rmse = float(df_comp.iloc[0]["test_rmse"])
    best_mae = float(df_comp.iloc[0]["test_mae"])

    logger.info(
        f"Model Comparison Result: Best model is '{best_algorithm}' "
        f"(R²={best_r2:.4f}, RMSE=${best_rmse:,.2f}, MAE=${best_mae:,.2f})"
    )

    if save_outputs:
        METRICS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        BEST_MODEL_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Save CSV
        df_comp.to_csv(METRICS_CSV_PATH, index=False)

        # Save JSON
        export_data = {
            "models": model_results,
            "comparison": df_comp.to_dict(orient="records"),
            "best_model": {
                "key": best_model_key,
                "algorithm": best_algorithm,
                "selection_criterion": "Highest Test R² with lowest RMSE/MAE",
                "test_r2": best_r2,
                "test_rmse": best_rmse,
                "test_mae": best_mae,
            },
            "metric_methodology_note": (
                "Continuous regression evaluation strictly utilizes R² (variance explained), "
                "MAE (dollar deviation), and RMSE (penalized large errors). Classification metrics "
                "like Accuracy, Precision, Recall, and F1-score are mathematically inapplicable."
            ),
        }
        with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        # Save best model metadata
        with open(BEST_MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(export_data["best_model"], f, indent=2)

        logger.info(f"Saved evaluation metrics to: {METRICS_JSON_PATH} and {METRICS_CSV_PATH}")

    return df_comp, best_model_key


def format_metrics_table(df_comp: pd.DataFrame) -> str:
    """Format comparison dataframe into a clean ASCII table for console display.

    Args:
        df_comp: Model comparison dataframe.

    Returns:
        Formatted multi-line string.
    """
    lines = [
        "",
        "=" * 78,
        f"{'MODEL COMPARISON SUMMARY':^78}",
        "=" * 78,
        f"{'Algorithm':<26} | {'Test R²':>10} | {'Test MAE ($)':>14} | {'Test RMSE ($)':>15}",
        "-" * 78,
    ]
    for _, row in df_comp.iterrows():
        algo = str(row["algorithm"])[:26]
        r2 = f"{row['test_r2']:.4f}"
        mae = f"${row['test_mae']:,.2f}"
        rmse = f"${row['test_rmse']:,.2f}"
        lines.append(f"{algo:<26} | {r2:>10} | {mae:>14} | {rmse:>15}")
    lines.append("=" * 78)
    return "\n".join(lines)
