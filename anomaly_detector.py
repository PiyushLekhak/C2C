import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from event_logger import get_logger

logger = get_logger("anomaly_detector")


def detect_anomalies_with_knn(
    df, k=5, scale_method="standard", contamination=0.05, save_path="plots"
):
    """
    Detects anomalies using KNN-based distance method.
    Returns:
        df_post (pd.DataFrame): Same data with anomaly columns added
        anomaly_summary (dict): Summary info for reporting
        anomaly_report (pd.DataFrame): Full score + is_anomaly table
    """
    logger.info("🔍 Starting KNN-based anomaly detection...")

    df_numeric = df.select_dtypes(include=[np.number]).copy()

    # === Scale numeric data ===
    if scale_method == "standard":
        scaler = StandardScaler()
    elif scale_method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scale_method. Choose 'standard' or 'minmax'.")

    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_numeric), columns=df_numeric.columns
    )

    # === KNN distances ===
    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(df_scaled)
    distances, _ = nbrs.kneighbors(df_scaled)
    scores = distances[:, 1:].mean(axis=1)

    # === Determine threshold ===
    threshold = np.percentile(scores, 100 * (1 - contamination))
    anomaly_flags = scores > threshold

    # === Save score distribution plot ===
    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, "anomaly_score_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(scores, kde=True, bins=30)
    plt.axvline(threshold, color="red", linestyle="--", label="Anomaly Threshold")
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"📊 Anomaly score distribution plot saved to {plot_path}")

    # === Add columns to original dataframe ===
    df_post = df.copy()
    df_post["anomaly_score"] = scores
    df_post["is_anomaly"] = anomaly_flags

    anomaly_summary = {
        "total_anomalies_flagged": int(anomaly_flags.sum()),
        "contamination_rate": contamination,
        "anomaly_plot_path": os.path.relpath(plot_path, start="reports"),
    }

    anomaly_report = pd.DataFrame(
        {"anomaly_score": scores, "is_anomaly": anomaly_flags}, index=df.index
    )

    logger.info(
        f"✅ Anomaly detection complete. {anomaly_flags.sum()} anomalies flagged."
    )

    return df_post, anomaly_summary, anomaly_report
