import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional

import joblib

VERSION_PATTERN = re.compile(r"^v(\d+)$")
CURRENT_FILE = "current.json"


def list_versions(model_dir) -> list:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return []
    found = [entry.name for entry in model_dir.iterdir() if entry.is_dir() and VERSION_PATTERN.match(entry.name)]
    return sorted(found, key=lambda name: int(name[1:]))


def next_version(model_dir) -> str:
    versions = list_versions(model_dir)
    return f"v{int(versions[-1][1:]) + 1}" if versions else "v1"


def save_version(model_dir, version: str, model, preprocessor, metadata: dict) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    final = model_dir / version
    if final.exists():
        raise FileExistsError(f"{final} already exists")

    temp = model_dir / f".{version}.tmp"
    shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir()
    joblib.dump(model, temp / "model.joblib")
    joblib.dump(preprocessor, temp / "preprocessor.joblib")
    (temp / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temp.rename(final)
    return final


def set_current(model_dir, version: str) -> None:
    model_dir = Path(model_dir)
    temp = model_dir / f".{CURRENT_FILE}.tmp"
    temp.write_text(json.dumps({"version": version}), encoding="utf-8")
    os.replace(temp, model_dir / CURRENT_FILE)


def current_version(model_dir) -> Optional[str]:
    path = Path(model_dir) / CURRENT_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("version")
    except (OSError, ValueError, AttributeError):
        return None
