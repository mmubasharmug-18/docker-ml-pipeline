import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

SCORES = ("accuracy", "precision", "recall", "f1", "roc_auc")


def cross_validated_scores(estimator, X, y, folds: int, random_state: int) -> dict:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    result = cross_validate(estimator, X, y, cv=splitter, scoring=list(SCORES))
    return {name: round(float(np.mean(result[f"test_{name}"])), 4) for name in SCORES}


def holdout_scores(model, X, y, threshold: float = 0.5) -> dict:
    probability = model.predict_proba(X)[:, 1]
    predicted = (probability >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        "precision": round(float(precision_score(y, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, probability)), 4),
    }


def confusion_counts(model, X, y, threshold: float = 0.5) -> dict:
    predicted = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def pick_best(scores: dict, metric: str) -> str:
    best_name = None
    best_value = None
    for name, values in scores.items():
        if best_value is None or values[metric] > best_value:
            best_name = name
            best_value = values[metric]
    return best_name
