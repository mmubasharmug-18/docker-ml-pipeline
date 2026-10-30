import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import sklearn

from serving.schemas import FIELD_NAMES

logger = logging.getLogger(__name__)

VERSION_PATTERN = re.compile(r"^v(\d+)$")
CURRENT_FILE = "current.json"
REQUIRED_FILES = ("model.joblib", "preprocessor.joblib", "metadata.json")


class ModelNotAvailable(Exception):
    pass


@dataclass
class LoadedModel:
    version: str
    model: object
    preprocessor: object
    metadata: dict
    sklearn_matches: bool

    @property
    def name(self) -> str:
        return self.metadata["model_name"]

    @property
    def features(self) -> list:
        return self.metadata["features"]

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("threshold", 0.5))

    @property
    def labels(self) -> dict:
        return self.metadata.get("labels", {"0": "negative", "1": "positive"})


def list_versions(model_dir) -> list:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return []
    found = [entry.name for entry in model_dir.iterdir() if entry.is_dir() and VERSION_PATTERN.match(entry.name)]
    return sorted(found, key=lambda name: int(name[1:]))


def current_version(model_dir) -> Optional[str]:
    try:
        return json.loads((Path(model_dir) / CURRENT_FILE).read_text(encoding="utf-8")).get("version")
    except (OSError, ValueError, AttributeError):
        return None


def load_model(model_dir, version: Optional[str] = None) -> LoadedModel:
    model_dir = Path(model_dir)
    chosen = version or current_version(model_dir)
    if not chosen:
        raise ModelNotAvailable(f"No trained model found in {model_dir}. Run the training step first.")

    folder = model_dir / chosen
    if not folder.is_dir():
        known = ", ".join(list_versions(model_dir)) or "none"
        raise ModelNotAvailable(f"Model version {chosen} does not exist in {model_dir} (available: {known})")

    missing = [name for name in REQUIRED_FILES if not (folder / name).is_file()]
    if missing:
        raise ModelNotAvailable(f"Model version {chosen} is incomplete, missing: {', '.join(missing)}")

    try:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        model = joblib.load(folder / "model.joblib")
        preprocessor = joblib.load(folder / "preprocessor.joblib")
    except Exception as error:
        raise ModelNotAvailable(f"Could not load model version {chosen}: {error}")

    features = metadata.get("features")
    if not isinstance(features, list) or set(features) != set(FIELD_NAMES):
        raise ModelNotAvailable(
            f"Model version {chosen} was trained on {features}, but the API expects {FIELD_NAMES}"
        )

    trained_with = metadata.get("libraries", {}).get("scikit_learn")
    matches = trained_with == sklearn.__version__
    if not matches:
        logger.warning(
            "Model %s was trained with scikit-learn %s but this service runs %s", chosen, trained_with, sklearn.__version__
        )

    return LoadedModel(version=chosen, model=model, preprocessor=preprocessor, metadata=metadata, sklearn_matches=matches)
