import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.utils.multiclass import type_of_target
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from event_logger import get_logger

logger = get_logger(module_name="model_evaluator")


def evaluate_model(y_true, y_pred):
    target_type = type_of_target(y_true)

    if target_type in ["binary", "multiclass"]:
        return evaluate_classification(y_true, y_pred)
    elif target_type in ["continuous", "continuous-multioutput"]:
        return evaluate_regression(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported target type: {target_type}")


def evaluate_classification(y_true, y_pred):
    metrics = {
        "Weighted F1 Score": f1_score(y_true, y_pred, average="weighted"),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist(),
        "Per-Class Metrics Report": classification_report(
            y_true, y_pred, output_dict=True
        ),
    }
    logger.log("Classification evaluation completed.")
    return metrics


def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "RMSE": rmse,
        "R² Score": r2,
    }
    logger.log("Regression evaluation completed.")
    return metrics


def evaluate_with_random_forest(
    X_raw, y_raw, X_clean, y_clean, test_size=0.2, random_state=42
):
    target_type = type_of_target(y_raw)

    if target_type in ["binary", "multiclass"]:
        ModelClass = RandomForestClassifier
        evaluator = evaluate_classification
    elif target_type in ["continuous", "continuous-multioutput"]:
        ModelClass = RandomForestRegressor
        evaluator = evaluate_regression
    else:
        raise ValueError(f"Unsupported target type: {target_type}")

    # Raw data
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=random_state
    )
    model_raw = ModelClass(random_state=random_state)
    model_raw.fit(X_train_raw, y_train_raw)
    y_pred_raw = model_raw.predict(X_test_raw)
    raw_metrics = evaluator(y_test_raw, y_pred_raw)

    # Clean data
    X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state
    )
    model_clean = ModelClass(random_state=random_state)
    model_clean.fit(X_train_clean, y_train_clean)
    y_pred_clean = model_clean.predict(X_test_clean)
    clean_metrics = evaluator(y_test_clean, y_pred_clean)

    logger.log("Model evaluation on raw and cleaned data complete.")
    return {
        "Raw Data Evaluation": raw_metrics,
        "Cleaned Data Evaluation": clean_metrics,
    }
