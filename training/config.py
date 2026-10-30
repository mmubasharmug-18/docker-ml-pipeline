import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

METRICS = ("roc_auc", "f1", "recall", "precision", "accuracy")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TrainSettings:
    data_path: Path = Path("data/diabetes.csv")
    model_dir: Path = Path("models")
    selection_metric: str = "roc_auc"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5


def read_number(env, name, default, convert, low, high, problems):
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = convert(raw)
    except ValueError:
        problems.append(f"{name} must be a number (got {raw!r})")
        return default
    if not low <= value <= high:
        problems.append(f"{name} must be between {low} and {high} (got {raw})")
        return default
    return value


def load_settings(env: Optional[Mapping[str, str]] = None) -> TrainSettings:
    env = os.environ if env is None else env
    problems = []

    raw_metric = env.get("SELECTION_METRIC", "").strip()
    metric = (raw_metric or "roc_auc").lower()
    if metric not in METRICS:
        problems.append(f"SELECTION_METRIC must be one of {', '.join(METRICS)} (got {raw_metric!r})")

    test_size = read_number(env, "TEST_SIZE", 0.2, float, 0.1, 0.5, problems)
    random_state = read_number(env, "RANDOM_STATE", 42, int, 0, 1_000_000, problems)
    cv_folds = read_number(env, "CV_FOLDS", 5, int, 2, 10, problems)

    if problems:
        raise ConfigError("; ".join(problems))

    return TrainSettings(
        data_path=Path(env.get("DATA_PATH", "").strip() or "data/diabetes.csv"),
        model_dir=Path(env.get("MODEL_DIR", "").strip() or "models"),
        selection_metric=metric,
        test_size=test_size,
        random_state=random_state,
        cv_folds=cv_folds,
    )
