import os
import subprocess
import sys
from pathlib import Path

from training.registry import current_version, list_versions

ROOT = Path(__file__).resolve().parents[2]


def run_training(**env):
    return subprocess.run(
        [sys.executable, "-m", "training"],
        env={**os.environ, "PYTHONUNBUFFERED": "1", **env},
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_command_trains_and_saves_a_version(tmp_path, dataset_csv):
    done = run_training(DATA_PATH=str(dataset_csv), MODEL_DIR=str(tmp_path), CV_FOLDS="3")
    assert done.returncode == 0
    assert "Selected:" in done.stdout
    assert "Saved as version v1" in done.stdout
    assert list_versions(tmp_path) == ["v1"]
    assert current_version(tmp_path) == "v1"


def test_missing_dataset_exits_with_code_2(tmp_path):
    done = run_training(DATA_PATH=str(tmp_path / "nope.csv"), MODEL_DIR=str(tmp_path / "models"))
    assert done.returncode == 2
    assert "Data problem" in done.stderr
    assert not (tmp_path / "models").exists()


def test_bad_setting_exits_with_code_2(tmp_path, dataset_csv):
    done = run_training(DATA_PATH=str(dataset_csv), MODEL_DIR=str(tmp_path), SELECTION_METRIC="LOUD")
    assert done.returncode == 2
    assert "SELECTION_METRIC" in done.stderr
