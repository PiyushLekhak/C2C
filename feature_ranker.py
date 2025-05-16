import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from event_logger import get_logger

logger = get_logger("feature_ranker")


def rank_features(
    X,
    y,
    task_type="classification",
    top_n=10,
    save_dir="plots",
    plot_name="feature_importance_after.png",
):
    """
    Ranks features using Random Forest importance.

    Args:
        X (pd.DataFrame): Cleaned feature set (no NaNs).
        y (pd.Series): Cleaned target variable.
        task_type (str): "classification" or "regression".
        top_n (int): Number of top features to visualize.
        save_dir (str): Folder to save plot in.
        plot_name (str): Filename for saved plot.

    Returns:
        Tuple:
            - pd.Series: Full sorted feature importances
            - list: Top N important features
            - str: Path to saved plot
    """
    df = X.copy()
    target = y.copy()

    # Drop rows with missing target values
    df["__target__"] = target
    df = df.dropna(subset=["__target__"])
    target = df.pop("__target__")

    # === Encode categorical target ===
    if target.dtype == "object" or target.dtype.name == "category":
        le = LabelEncoder()
        target = le.fit_transform(target)

    # === Encode categorical features ===
    df = pd.get_dummies(df, drop_first=True)

    # === Choose model type ===
    if task_type == "classification":
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif task_type == "regression":
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")

    # === Train & rank ===
    model.fit(df, target)
    importances = pd.Series(model.feature_importances_, index=df.columns)
    importances_sorted = importances.sort_values(ascending=False)

    top_features = importances_sorted.head(top_n).index.tolist()

    # === Save plot ===
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, plot_name)

    plt.figure(figsize=(10, 6))
    importances_sorted.head(top_n).plot(kind="bar", color="skyblue")
    plt.title("Top Feature Importances")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"📊 Feature importance plot saved to {plot_path}")
    logger.info(f"🏅 Top {top_n} features: {top_features}")

    return importances_sorted, top_features, plot_path
