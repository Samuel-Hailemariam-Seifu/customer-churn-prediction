from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray, thresholds: List[float]) -> Dict:
    scored = [evaluate_threshold(y_true, y_proba, t) for t in thresholds]
    best = max(scored, key=lambda x: (x["f1"], x["recall"], x["precision"]))
    return {"best": best, "all": scored}
