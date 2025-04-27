import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
from event_logger import get_logger

logger = get_logger("feature_ranker")


def rank_features(
    X,
    y,
    task_type="regression",
    save_dir="plots",
    plot_name="feature_importance.png",
):
    """
    Ranks features based on their importance using a Random Forest model.
    Now returns importance scores AND saved plot path.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        task_type (str): "classification" or "regression".
        save_dir (str): Directory to save plots.
        plot_name (str): Name for the output plot.

    Returns:
        (pd.Series, str): (Sorted feature importances, path to saved plot)
    """
    # Handle categorical targets
    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        y = le.fit_transform(y)

    if task_type == "classification":
        model = RandomForestClassifier(random_state=42)
    elif task_type == "regression":
        model = RandomForestRegressor(random_state=42)
    else:
        raise ValueError("Invalid task_type. Choose 'classification' or 'regression'.")

    if X.select_dtypes(include=["object", "category"]).shape[1] > 0:
        X = pd.get_dummies(X)

    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances_sorted = importances.sort_values(ascending=False)

    # Save feature importance plot
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, plot_name)

    plt.figure(figsize=(10, 6))
    importances_sorted.iloc[:7].plot(kind="bar")  # Show only top 7 features
    plt.title("Feature Importance Ranking")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"📊 Feature importance plot saved to {plot_path}")

    return importances_sorted, plot_path
