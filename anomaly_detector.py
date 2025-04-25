import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from event_logger import get_logger

logger = get_logger(module_name="anomaly_detector")


def detect_anomalies_with_knn(
    df,
    k=5,
    scale_method=None,
    contamination=0.05,
    post_action="none",
    cap_quantiles=(0.01, 0.99),
):
    """
    Detects anomalies using a K-Nearest Neighbors distance-based method.
    Includes optional preprocessing and lightweight post-cleaning logic.

    Args:
        df (pd.DataFrame): Cleaned numeric DataFrame.
        k (int): Number of neighbors to consider (excluding self).
        scale_method (str): Optional feature scaling: 'standard', 'minmax', or None.
        contamination (float): Proportion of data to flag as anomalies.
        post_action (str): Post-detection action: 'none', 'remove', or 'cap'.
        cap_quantiles (tuple): Quantile bounds used when capping (if applicable).

    Returns:
        pd.DataFrame: Post-processed DataFrame after optional anomaly cleaning.
        pd.DataFrame: Anomaly report with scores and flags.
    """
    logger.log("🔍 Starting KNN-based anomaly detection...")

    df_numeric = df.select_dtypes(include=[np.number]).copy()
    df_scaled = df_numeric.copy()

    if scale_method == "standard":
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_numeric), columns=df_numeric.columns
        )
        logger.log("✅ Standard scaling applied.")
    elif scale_method == "minmax":
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_numeric), columns=df_numeric.columns
        )
        logger.log("✅ Min-max scaling applied.")

    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(df_scaled)
    distances, _ = nbrs.kneighbors(df_scaled)
    scores = distances[:, 1:].mean(axis=1)

    threshold = np.percentile(scores, 100 * (1 - contamination))
    anomaly_flags = scores > threshold

    anomaly_report = pd.DataFrame(
        {"anomaly_score": scores, "is_anomaly": anomaly_flags}, index=df.index
    )

    df_post = df.copy()
    if post_action == "remove":
        df_post = df[~anomaly_flags].copy()
        logger.log("⚠️ Anomalous rows removed.")
    elif post_action == "cap":
        for col in df_numeric.columns:
            lower, upper = df[col].quantile(cap_quantiles[0]), df[col].quantile(
                cap_quantiles[1]
            )
            df_post.loc[anomaly_flags, col] = np.clip(
                df_post.loc[anomaly_flags, col], lower, upper
            )
        logger.log(f"⚠️ Anomalous values capped using quantiles {cap_quantiles}.")

    logger.log(
        f"🎯 Anomaly detection complete. {anomaly_flags.sum()} anomalies flagged."
    )
    return df_post, anomaly_report
