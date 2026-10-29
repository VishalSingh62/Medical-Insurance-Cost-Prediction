"""Data Visualization and Model Performance Plotting Module.

Generates 8 high-resolution publication-quality figures:
1. Distribution of Insurance Charges
2. Age vs Insurance Charges (with Smoker stratification)
3. BMI vs Insurance Charges (with Obesity Threshold)
4. Charges by Smoker Status (Violin & Box Plot)
5. Numerical Features Correlation Heatmap
6. Actual vs Predicted Charges (Side-by-Side Model Comparison)
7. Residual and Error Distribution
8. Model Performance Metrics Comparison (R², MAE, RMSE)
"""

from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless chart generation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import OUTPUTS_DIR, logger
from src.data_loader import load_csv_data
from src.preprocessing import clean_and_preprocess_dataset, prepare_train_test_split


def set_plot_style() -> None:
    """Configure modern Seaborn and Matplotlib theme settings."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update(
        {
            "figure.autolayout": True,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 1.2,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "semibold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.family": "sans-serif",
        }
    )


def plot_charges_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Chart 1: Distribution and Kernel Density of Insurance Charges."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    mean_val = df["charges"].mean()
    median_val = df["charges"].median()

    sns.histplot(
        df["charges"],
        kde=True,
        color="#2563EB",
        bins=35,
        ax=ax,
        edgecolor="#1E40AF",
        alpha=0.6,
    )
    ax.axvline(mean_val, color="#EF4444", linestyle="--", linewidth=2, label=f"Mean: ${mean_val:,.2f}")
    ax.axvline(median_val, color="#10B981", linestyle="-.", linewidth=2, label=f"Median: ${median_val:,.2f}")

    ax.set_title("Distribution of Medical Insurance Charges", pad=12)
    ax.set_xlabel("Insurance Charges ($)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", frameon=True)

    save_path = output_dir / "01_charges_distribution.png"
    plt.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_age_vs_charges(df: pd.DataFrame, output_dir: Path) -> Path:
    """Chart 2: Age vs Insurance Charges stratified by Smoker Status."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    palette = {"yes": "#EF4444", "no": "#10B981"}
    sns.scatterplot(
        data=df,
        x="age",
        y="charges",
        hue="smoker",
        palette=palette,
        alpha=0.75,
        s=55,
        ax=ax,
    )
    # Regression trendlines for both smoker groups
    sns.regplot(
        data=df[df["smoker"] == "yes"],
        x="age",
        y="charges",
        scatter=False,
        ax=ax,
        color="#DC2626",
        line_kws={"linewidth": 2, "label": "Smoker Trend"},
    )
    sns.regplot(
        data=df[df["smoker"] == "no"],
        x="age",
        y="charges",
        scatter=False,
        ax=ax,
        color="#059669",
        line_kws={"linewidth": 2, "label": "Non-Smoker Trend"},
    )

    ax.set_title("Age vs. Insurance Charges (Grouped by Smoking Status)", pad=12)
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Charges ($)")
    ax.legend(title="Smoker Status", frameon=True)

    save_path = output_dir / "02_age_vs_charges.png"
    plt.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_bmi_vs_charges(df: pd.DataFrame, output_dir: Path) -> Path:
    """Chart 3: BMI vs Insurance Charges highlighting Clinical Obesity Threshold (BMI 30)."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    palette = {"yes": "#EF4444", "no": "#3B82F6"}
    sns.scatterplot(
        data=df,
        x="bmi",
        y="charges",
        hue="smoker",
        palette=palette,
        alpha=0.75,
        s=55,
        ax=ax,
    )
    # Clinical Obesity Threshold line
    ax.axvline(30.0, color="#D97706", linestyle=":", linewidth=2.5, label="Obesity Threshold (BMI ≥ 30)")

    ax.set_title("BMI vs. Insurance Charges (Interaction with Smoking)", pad=12)
    ax.set_xlabel("Body Mass Index (BMI)")
    ax.set_ylabel("Charges ($)")
    ax.legend(title="Smoker Status", frameon=True)

    save_path = output_dir / "03_bmi_vs_charges.png"
    plt.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_charges_by_smoker(df: pd.DataFrame, output_dir: Path) -> Path:
    """Chart 4: Box and Violin plot of Charges by Smoker Status."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    palette = {"yes": "#EF4444", "no": "#10B981"}

    # Box Plot
    sns.boxplot(
        data=df,
        x="smoker",
        y="charges",
        hue="smoker",
        palette=palette,
        ax=ax1,
        width=0.45,
        legend=False,
        boxprops=dict(alpha=0.8),
    )
    ax1.set_title("Box Plot: Charges by Smoker Status", pad=10)
    ax1.set_xlabel("Smoker Status")
    ax1.set_ylabel("Charges ($)")

    # Violin Plot
    sns.violinplot(
        data=df,
        x="smoker",
        y="charges",
        hue="smoker",
        palette=palette,
        ax=ax2,
        cut=0,
        inner="quartile",
        legend=False,
        alpha=0.8,
    )

    ax2.set_title("Violin Plot: Density Distribution by Smoker", pad=10)
    ax2.set_xlabel("Smoker Status")
    ax2.set_ylabel("Charges ($)")

    save_path = output_dir / "04_charges_by_smoker.png"
    plt.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Chart 5: Correlation Heatmap for Numerical Features."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    num_cols = ["age", "bmi", "children", "charges"]
    corr = df[num_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=-0.2,
        vmax=1.0,
        square=True,
        linewidths=1.2,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Pearson Correlation Heatmap (Numerical Variables)", pad=12)

    save_path = output_dir / "05_correlation_heatmap.png"
    plt.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_actual_vs_predicted(
    y_test: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_dir: Path,
) -> Path:
    """Chart 6: Actual vs Predicted Scatter Plots with Perfect Fit Identity Line."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

    min_val = min(y_test.min(), min(p.min() for p in predictions.values()))
    max_val = max(y_test.max(), max(p.max() for p in predictions.values()))

    colors = {"linear_regression": "#3B82F6", "random_forest": "#10B981"}
    titles = {
        "linear_regression": "Linear Regression: Actual vs. Predicted",
        "random_forest": "Random Forest Regressor: Actual vs. Predicted",
    }

    for idx, (key, y_pred) in enumerate(predictions.items()):
        ax = axes[idx]
        ax.scatter(y_test, y_pred, color=colors.get(key, "#6366F1"), alpha=0.6, s=40, edgecolors="none")
        # 45-degree reference line
        ax.plot([min_val, max_val], [min_val, max_val], color="#EF4444", linestyle="--", linewidth=2, label="Perfect Prediction (y=x)")
        ax.set_title(titles.get(key, key), pad=10)
        ax.set_xlabel("Actual Charges ($)")
        ax.set_ylabel("Predicted Charges ($)")
        ax.set_xlim(min_val * 0.9, max_val * 1.05)
        ax.set_ylim(min_val * 0.9, max_val * 1.05)
        ax.legend(loc="upper left", frameon=True)

    save_path = output_dir / "06_actual_vs_predicted.png"
    plt.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_residual_distribution(
    y_test: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_dir: Path,
) -> Path:
    """Chart 7: Residual / Error Distribution for both algorithms."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    colors = {"linear_regression": "#3B82F6", "random_forest": "#10B981"}

    for idx, (key, y_pred) in enumerate(predictions.items()):
        residuals = y_test - y_pred
        ax = axes[idx]
        sns.histplot(
            residuals,
            kde=True,
            color=colors.get(key, "#6366F1"),
            bins=30,
            ax=ax,
            alpha=0.6,
        )
        ax.axvline(0, color="#EF4444", linestyle="--", linewidth=2, label="Zero Error")
        algo_name = "Linear Regression" if key == "linear_regression" else "Random Forest"
        ax.set_title(f"{algo_name} Residuals (Actual - Predicted)", pad=10)
        ax.set_xlabel("Residual ($)")
        ax.set_ylabel("Frequency")
        ax.legend(loc="upper right", frameon=True)

    save_path = output_dir / "07_residual_distribution.png"
    plt.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_model_metrics_comparison(
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Chart 8: Model Performance Comparison Bar Charts (R², MAE, RMSE)."""
    set_plot_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

    # Colors for algorithms
    bar_colors = ["#3B82F6", "#10B981"]

    # R2 Plot
    sns.barplot(data=metrics_df, x="algorithm", y="test_r2", hue="algorithm", palette=bar_colors, ax=ax1, width=0.45, legend=False)
    ax1.set_title("Test R² Score (Higher is Better)", pad=10)
    ax1.set_xlabel("")
    ax1.set_ylabel("R²")
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2.0, p.get_height() / 2.0),
                     ha="center", va="center", color="white", weight="bold", fontsize=11)

    # MAE Plot
    sns.barplot(data=metrics_df, x="algorithm", y="test_mae", hue="algorithm", palette=bar_colors, ax=ax2, width=0.45, legend=False)
    ax2.set_title("Test MAE ($) (Lower is Better)", pad=10)
    ax2.set_xlabel("")
    ax2.set_ylabel("MAE ($)")
    for p in ax2.patches:
        ax2.annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2.0, p.get_height() / 2.0),
                     ha="center", va="center", color="white", weight="bold", fontsize=11)

    # RMSE Plot
    sns.barplot(data=metrics_df, x="algorithm", y="test_rmse", hue="algorithm", palette=bar_colors, ax=ax3, width=0.45, legend=False)
    ax3.set_title("Test RMSE ($) (Lower is Better)", pad=10)
    ax3.set_xlabel("")
    ax3.set_ylabel("RMSE ($)")
    for p in ax3.patches:
        ax3.annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2.0, p.get_height() / 2.0),
                     ha="center", va="center", color="white", weight="bold", fontsize=11)


    save_path = output_dir / "08_model_metrics_comparison.png"
    plt.savefig(save_path)
    plt.close(fig)
    return save_path


def generate_all_visualizations(
    df: Optional[pd.DataFrame] = None,
    pipelines: Optional[Dict[str, Pipeline]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Generate and save all 8 evaluation and exploratory charts.

    Args:
        df: Cleaned DataFrame. If None, loads cleaned/raw dataset automatically.
        pipelines: Dict of trained models. If None, loads saved joblib pipelines.
        output_dir: Path to directory for saving images. Defaults to OUTPUTS_DIR.

    Returns:
        Dictionary of generated image file paths.
    """
    target_dir = Path(output_dir or OUTPUTS_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Load data if needed
    if df is None:
        raw_df = load_csv_data()
        clean_df, _ = clean_and_preprocess_dataset(raw_df, save_cleaned=False)
    else:
        clean_df = df

    # Prepare train/test sets
    X_train, X_test, y_train, y_test = prepare_train_test_split(clean_df)

    # Load pipelines if not provided
    if pipelines is None:
        import joblib
        from src.config import LINEAR_REGRESSION_MODEL_PATH, RANDOM_FOREST_MODEL_PATH
        if not LINEAR_REGRESSION_MODEL_PATH.exists() or not RANDOM_FOREST_MODEL_PATH.exists():
            from src.train_models import train_and_evaluate_all_models
            trained_pipes, _, _ = train_and_evaluate_all_models()
            pipelines = trained_pipes
        else:
            pipelines = {
                "linear_regression": joblib.load(LINEAR_REGRESSION_MODEL_PATH),
                "random_forest": joblib.load(RANDOM_FOREST_MODEL_PATH),
            }

    # Generate predictions on test set
    preds = {key: pipe.predict(X_test) for key, pipe in pipelines.items()}

    # Compute comparison df
    from src.evaluation import compute_regression_metrics, compare_models
    results = {}
    for k, p in preds.items():
        name = "Linear Regression" if k == "linear_regression" else "Random Forest"
        results[k] = compute_regression_metrics(y_test, p, model_name=name)
    df_comp, _ = compare_models(results, save_outputs=False)

    logger.info("Generating publication-quality charts...")
    generated_charts = {
        "charges_distribution": plot_charges_distribution(clean_df, target_dir),
        "age_vs_charges": plot_age_vs_charges(clean_df, target_dir),
        "bmi_vs_charges": plot_bmi_vs_charges(clean_df, target_dir),
        "charges_by_smoker": plot_charges_by_smoker(clean_df, target_dir),
        "correlation_heatmap": plot_correlation_heatmap(clean_df, target_dir),
        "actual_vs_predicted": plot_actual_vs_predicted(y_test.values, preds, target_dir),
        "residual_distribution": plot_residual_distribution(y_test.values, preds, target_dir),
        "metrics_comparison": plot_model_metrics_comparison(df_comp, target_dir),
    }

    logger.info(f"All 8 visualization figures saved to: {target_dir}")
    return generated_charts


if __name__ == "__main__":
    generate_all_visualizations()
