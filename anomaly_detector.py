import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from event_logger import get_logger

logger = get_logger("anomaly_detector")


def detect_anomalies_with_knn(
    df,
    k=5,
    scale_method=None,
    contamination=0.05,
    post_action="none",
    cap_quantiles=(0.01, 0.99),
    save_path="plots",
):
    """
    Detects anomalies using KNN-based distance method.
    Now includes score plot saving + summary info for report.

    Returns:
        df_post (pd.DataFrame): Cleaned or modified dataset.
        anomaly_report (pd.DataFrame): Scores + is_anomaly flags.
        summary (dict): Summary info for reporting.
    """
    logger.info("🔍 Starting KNN-based anomaly detection...")

    df_numeric = df.select_dtypes(include=[np.number]).copy()
    df_scaled = df_numeric.copy()

    if scale_method == "standard":
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_numeric), columns=df_numeric.columns
        )
        logger.info("✅ Standard scaling applied.")
    elif scale_method == "minmax":
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_numeric), columns=df_numeric.columns
        )
        logger.info("✅ Min-max scaling applied.")

    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(df_scaled)
    distances, _ = nbrs.kneighbors(df_scaled)
    scores = distances[:, 1:].mean(axis=1)

    threshold = np.percentile(scores, 100 * (1 - contamination))
    anomaly_flags = scores > threshold

    anomaly_report = pd.DataFrame(
        {"anomaly_score": scores, "is_anomaly": anomaly_flags}, index=df.index
    )

    # === Plot score distribution ===
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

    # === Post-action on anomalies ===
    df_post = df.copy()
    action_taken = "none"

    if post_action == "remove":
        df_post = df[~anomaly_flags].copy()
        action_taken = "removed"
        logger.info("⚠️ Anomalous rows removed.")

    elif post_action == "cap":
        for col in df_numeric.columns:
            lower, upper = df[col].quantile(cap_quantiles[0]), df[col].quantile(
                cap_quantiles[1]
            )
            df_post.loc[anomaly_flags, col] = np.clip(
                df_post.loc[anomaly_flags, col], lower, upper
            )
        action_taken = "capped"
        logger.info(f"⚠️ Anomalous values capped using quantiles {cap_quantiles}.")

    logger.info(
        f"🎯 Anomaly detection complete. {anomaly_flags.sum()} anomalies flagged."
    )

    summary = {
        "total_anomalies_flagged": int(anomaly_flags.sum()),
        "post_action": action_taken,
        "anomaly_plot_path": plot_path,
        "contamination_rate": contamination,
    }

    return df_post, anomaly_report, summary
