import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
from event_logger import get_logger

logger = get_logger(module_name="feature_ranker")


def rank_features(X, y, task_type="classification", save_path="feature_importance.png"):
    """
    Ranks features based on their importance using a Random Forest model.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        task_type (str): Either "classification" or "regression".
        save_path (str): Path to save the feature importance plot.

    Returns:
        pd.Series: Sorted feature importance scores.
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

    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances_sorted = importances.sort_values(ascending=False)

    # Save plot
    plt.figure(figsize=(10, 6))
    importances_sorted.plot(kind="bar")
    plt.title("Feature Importance Ranking")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    logger.log(f"Feature importance plot saved to {save_path}")
    return importances_sorted
