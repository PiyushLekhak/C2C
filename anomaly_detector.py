import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from event_logger import get_logger

logger = get_logger("anomaly_detector")


def detect_anomalies_with_isolation_forest(
    df,
    contamination=0.05,
    save_path="plots",
    random_state=42,
    target_column=None,
    target_values=None,
):
    """
    Detect anomalies using Isolation Forest with basic categorical encoding.
    Returns:
        df_post (pd.DataFrame): Original data with 'anomaly_score' and 'is_anomaly'
        anomaly_summary (dict): Metadata for reporting
        anomaly_report (pd.DataFrame): Table with scores + anomaly flags
    """
    logger.info("🌲 Starting Isolation Forest-based anomaly detection...")

    # === Encode categoricals ===
    df_encoded = pd.get_dummies(df, drop_first=True)

    # === Handle missing values for fitting ===
    imputer = SimpleImputer(strategy="mean")
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_encoded), columns=df_encoded.columns, index=df.index
    )

    # === Isolation Forest ===
    iso = IsolationForest(
        contamination=contamination, random_state=random_state, n_jobs=-1, verbose=0
    )
    iso.fit(df_imputed)

    scores = -iso.decision_function(df_imputed)  # higher = more anomalous
    flags = iso.predict(df_imputed)  # -1 = anomaly, 1 = normal
    anomaly_flags = flags == -1

    # === Save distribution plot ===
    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, "anomaly_score_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(scores, bins=30, kde=True)
    plt.axvline(
        np.percentile(scores, 100 * (1 - contamination)),
        color="red",
        linestyle="--",
        label="Anomaly Threshold",
    )
    plt.title("Isolation Forest Anomaly Scores")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"📊 Anomaly score distribution plot saved to {plot_path}")

    # === Add to original DataFrame ===
    df_post = df.copy()
    df_post["anomaly_score"] = scores
    df_post["is_anomaly"] = anomaly_flags

    if target_column and target_values is not None:
        df_post[target_column] = target_values

    anomaly_summary = {
        "total_anomalies_flagged": int(anomaly_flags.sum()),
        "contamination_rate": contamination,
        "anomaly_plot_path": os.path.relpath(plot_path, start="reports"),
    }

    anomaly_report = df_post.copy()
    anomaly_report = anomaly_report.sort_values(by="anomaly_score", ascending=False)

    logger.info(f"✅ Isolation Forest flagged {anomaly_flags.sum()} anomalies.")

    return df_post, anomaly_summary, anomaly_report


def get_anomaly_flags(df, contamination=0.05, random_state=42):
    """
    Return a boolean Series indexed like df where True == anomaly.
    Uses mean-imputed + one-hot encoded copy internally so original df is never changed.
    """
    df_post, _, _ = detect_anomalies_with_isolation_forest(
        df.copy(),
        contamination=contamination,
        save_path="plots",
        random_state=random_state,
    )
    return df_post["is_anomaly"]
