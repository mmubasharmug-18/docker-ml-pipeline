import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from training.config import TrainSettings
from training.data import FEATURES, TARGET, file_hash, load_dataset
from training.evaluate import SCORES, confusion_counts, cross_validated_scores, holdout_scores, pick_best
from training.models import candidate_models
from training.preprocessing import build_preprocessor
from training.registry import next_version, save_version, set_current

THRESHOLD = 0.5
LABELS = {"0": "not diabetic", "1": "diabetic"}


def split_data(frame: pd.DataFrame, settings: TrainSettings):
    return train_test_split(
        frame[FEATURES],
        frame[TARGET],
        test_size=settings.test_size,
        stratify=frame[TARGET],
        random_state=settings.random_state,
    )


def library_versions() -> dict:
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }


def run(settings: TrainSettings, progress=None) -> dict:
    say = progress or (lambda text: None)

    say(f"Loading {settings.data_path}")
    frame = load_dataset(settings.data_path)
    X_train, X_test, y_train, y_test = split_data(frame, settings)
    say(f"{len(X_train)} rows to train on, {len(X_test)} rows held back for testing")

    candidates = candidate_models(settings.random_state)
    cv_scores = {}
    for name, estimator in candidates.items():
        say(f"Cross-validating {name}")
        full = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        cv_scores[name] = cross_validated_scores(full, X_train, y_train, settings.cv_folds, settings.random_state)

    best_name = pick_best(cv_scores, settings.selection_metric)
    say(f"Best by {settings.selection_metric}: {best_name}")

    preprocessor = build_preprocessor().fit(X_train)
    train_ready = preprocessor.transform(X_train)
    test_ready = preprocessor.transform(X_test)

    models = {}
    test_scores = {}
    for name, estimator in candidates.items():
        say(f"Training {name} and scoring it on the test set")
        models[name] = clone(estimator).fit(train_ready, y_train)
        test_scores[name] = holdout_scores(models[name], test_ready, y_test, THRESHOLD)

    best_model = models[best_name]
    version = next_version(settings.model_dir)

    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": best_name,
        "selection_metric": settings.selection_metric,
        "selection_score": cv_scores[best_name][settings.selection_metric],
        "threshold": THRESHOLD,
        "labels": LABELS,
        "features": FEATURES,
        "target": TARGET,
        "dataset": {
            "file": settings.data_path.name,
            "sha256": file_hash(settings.data_path),
            "rows": int(len(frame)),
            "positive_rate": round(float(frame[TARGET].mean()), 4),
        },
        "split": {
            "test_size": settings.test_size,
            "random_state": settings.random_state,
            "cv_folds": settings.cv_folds,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "candidates": [
            {"name": name, "cross_validation": cv_scores[name], "test": test_scores[name]} for name in candidates
        ],
        "test_metrics": test_scores[best_name],
        "confusion_matrix": confusion_counts(best_model, test_ready, y_test, THRESHOLD),
        "libraries": library_versions(),
    }

    save_version(settings.model_dir, version, best_model, preprocessor, metadata)
    set_current(settings.model_dir, version)
    say(f"Saved {version} in {settings.model_dir}")
    return metadata


def format_report(metadata: dict) -> str:
    header = f"{'Model':<22}" + "".join(f"{name:>11}" for name in SCORES)

    def table(key):
        lines = [header]
        for candidate in metadata["candidates"]:
            marker = " *" if candidate["name"] == metadata["model_name"] else ""
            row = f"{candidate['name']:<22}" + "".join(f"{candidate[key][name]:>11.4f}" for name in SCORES)
            lines.append(row + marker)
        return "\n".join(lines)

    data = metadata["dataset"]
    split = metadata["split"]
    cm = metadata["confusion_matrix"]
    return "\n".join(
        [
            f"Dataset: {data['file']}, {data['rows']} rows, {data['positive_rate'] * 100:.1f}% positive",
            f"Split: {split['train_rows']} train, {split['test_rows']} test",
            "",
            f"Cross-validation on the training rows (mean of {split['cv_folds']} folds)",
            table("cross_validation"),
            "",
            "Test set (rows the models never saw)",
            table("test"),
            "",
            f"Selected: {metadata['model_name']} (best {metadata['selection_metric']} in cross-validation: {metadata['selection_score']:.4f})",
            f"Test confusion matrix: {cm['tn']} true negative, {cm['fp']} false positive, {cm['fn']} false negative, {cm['tp']} true positive",
            f"Saved as version {metadata['version']}",
        ]
    )
