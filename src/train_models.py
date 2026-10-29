"""Model Training, Pipeline Encapsulation, and Persistence Module.

Builds Scikit-Learn Pipelines combining ColumnTransformer preprocessing
with Linear Regression and Random Forest Regressor models, evaluates them on
identical test splits, and persists artifacts using Joblib.
"""

from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.config import (
    LINEAR_REGRESSION_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    BEST_MODEL_PATH,
    RANDOM_STATE,
    logger,
)
from src.data_loader import load_csv_data
from src.preprocessing import (
    clean_and_preprocess_dataset,
    build_preprocessor,
    prepare_train_test_split,
)
from src.evaluation import (
    compute_regression_metrics,
    compare_models,
    format_metrics_table,
)


def build_linear_regression_pipeline() -> Pipeline:
    """Build Scikit-Learn Pipeline for Linear Regression.

    Encapsulates preprocessing and standard Ordinary Least Squares regression.

    Returns:
        Configured Pipeline instance.
    """
    preprocessor = build_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    return pipeline


def build_random_forest_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Build Scikit-Learn Pipeline for Random Forest Regressor.

    Encapsulates preprocessing and an ensemble of 100 decision trees.

    Args:
        random_state: Seed for tree sampling reproducibility.

    Returns:
        Configured Pipeline instance.
    """
    preprocessor = build_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


def train_and_evaluate_all_models() -> Tuple[Dict[str, Pipeline], pd.DataFrame, str]:
    """Train all regression models, evaluate on holdout test set, and persist pipelines.

    Execution Flow:
    1. Load raw data and run dataset cleaning & auditing.
    2. Split features and target into reproducible train (80%) and test (20%) sets.
    3. Fit Linear Regression pipeline on train set.
    4. Fit Random Forest pipeline on train set.
    5. Evaluate both on test set for R², MAE, RMSE.
    6. Compare metrics, pick best model, and persist all models using Joblib.

    Returns:
        Tuple of (trained_pipelines_dict, comparison_dataframe, best_model_key)
    """
    logger.info("Starting model training and evaluation workflow...")

    # Step 1: Load and clean data
    raw_df = load_csv_data()
    clean_df, _ = clean_and_preprocess_dataset(raw_df, save_cleaned=True)

    # Step 2: Leakage-free train/test partition
    X_train, X_test, y_train, y_test = prepare_train_test_split(clean_df)

    # Pipelines definition
    pipelines = {
        "linear_regression": {
            "name": "Linear Regression",
            "pipeline": build_linear_regression_pipeline(),
            "save_path": LINEAR_REGRESSION_MODEL_PATH,
        },
        "random_forest": {
            "name": "Random Forest Regressor",
            "pipeline": build_random_forest_pipeline(),
            "save_path": RANDOM_FOREST_MODEL_PATH,
        },
    }

    eval_results: Dict[str, Dict[str, Any]] = {}
    trained_pipelines: Dict[str, Pipeline] = {}

    for key, spec in pipelines.items():
        name = spec["name"]
        pipe = spec["pipeline"]
        save_path = spec["save_path"]

        logger.info(f"Training {name} pipeline on {len(X_train)} samples...")
        pipe.fit(X_train, y_train)

        # Predictions on train and test
        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)

        # Metrics computation
        train_metrics = compute_regression_metrics(y_train, y_train_pred, model_name=f"{name} (Train)")
        test_metrics = compute_regression_metrics(y_test, y_test_pred, model_name=name)

        eval_results[key] = {
            "model_name": name,
            "train_r2": train_metrics["r2_score"],
            "test_r2": test_metrics["r2_score"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_mape_percent": test_metrics["mape_percent"],
            "sample_count_train": len(y_train),
            "sample_count_test": len(y_test),
        }

        # Save individual model
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, save_path)
        logger.info(f"Saved {name} pipeline to: {save_path}")

        trained_pipelines[key] = pipe

    # Step 3: Compare models using test metrics
    df_comp, best_key = compare_models(eval_results, save_outputs=True)
    table_str = format_metrics_table(df_comp)
    print(table_str)

    # Step 4: Save best model pointer
    best_pipe = trained_pipelines[best_key]
    joblib.dump(best_pipe, BEST_MODEL_PATH)
    logger.info(f"Persisted best performing model ({best_key}) to: {BEST_MODEL_PATH}")

    return trained_pipelines, df_comp, best_key


if __name__ == "__main__":
    train_and_evaluate_all_models()
