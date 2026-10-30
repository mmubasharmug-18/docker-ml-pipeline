import json

import joblib
import pytest

from training.config import TrainSettings
from training.data import FEATURES, DataError, load_dataset
from training.evaluate import SCORES, holdout_scores
from training.pipeline import format_report, run, split_data
from training.registry import current_version, list_versions


def read_metadata(model_dir, version="v1"):
    return json.loads((model_dir / version / "metadata.json").read_text())


def test_first_run_creates_v1_and_marks_it_current(model_dir):
    assert list_versions(model_dir) == ["v1"]
    assert current_version(model_dir) == "v1"
    assert sorted(path.name for path in (model_dir / "v1").iterdir()) == [
        "metadata.json",
        "model.joblib",
        "preprocessor.joblib",
    ]


def test_metadata_describes_the_run(model_dir):
    meta = read_metadata(model_dir)
    assert meta["version"] == "v1"
    assert meta["features"] == FEATURES
    assert meta["dataset"]["rows"] == 300
    assert len(meta["dataset"]["sha256"]) == 64
    assert meta["split"]["train_rows"] + meta["split"]["test_rows"] == 300
    assert meta["libraries"]["scikit_learn"]
    assert meta["created_at"]


def test_at_least_two_models_were_compared_on_all_metrics(model_dir):
    candidates = read_metadata(model_dir)["candidates"]
    assert len(candidates) >= 2
    for candidate in candidates:
        assert set(candidate["cross_validation"]) == set(SCORES)
        assert set(candidate["test"]) == set(SCORES)


def test_selected_model_has_the_best_cross_validation_score(model_dir):
    meta = read_metadata(model_dir)
    metric = meta["selection_metric"]
    scores = {c["name"]: c["cross_validation"][metric] for c in meta["candidates"]}
    assert meta["model_name"] == max(scores, key=scores.get)
    assert meta["selection_score"] == scores[meta["model_name"]]


def test_saved_model_reproduces_the_reported_test_metrics(model_dir, dataset_csv):
    meta = read_metadata(model_dir)
    model = joblib.load(model_dir / "v1" / "model.joblib")
    preprocessor = joblib.load(model_dir / "v1" / "preprocessor.joblib")

    settings = TrainSettings(data_path=dataset_csv, model_dir=model_dir, cv_folds=3)
    _, X_test, _, y_test = split_data(load_dataset(dataset_csv), settings)
    again = holdout_scores(model, preprocessor.transform(X_test), y_test)

    assert again == meta["test_metrics"]


def test_saved_model_is_better_than_chance(model_dir):
    assert read_metadata(model_dir)["test_metrics"]["roc_auc"] > 0.7


def test_second_run_makes_v2_and_keeps_v1(two_version_dir):
    assert list_versions(two_version_dir) == ["v1", "v2"]
    assert current_version(two_version_dir) == "v2"
    assert read_metadata(two_version_dir, "v1")["version"] == "v1"
    assert read_metadata(two_version_dir, "v2")["version"] == "v2"


def test_selection_metric_can_be_changed(tmp_path, dataset_csv):
    settings = TrainSettings(data_path=dataset_csv, model_dir=tmp_path, cv_folds=3, selection_metric="recall")
    meta = run(settings)
    recalls = {c["name"]: c["cross_validation"]["recall"] for c in meta["candidates"]}
    assert meta["selection_metric"] == "recall"
    assert meta["model_name"] == max(recalls, key=recalls.get)


def test_same_settings_give_the_same_scores(tmp_path, dataset_csv):
    first = run(TrainSettings(data_path=dataset_csv, model_dir=tmp_path / "a", cv_folds=3))
    second = run(TrainSettings(data_path=dataset_csv, model_dir=tmp_path / "b", cv_folds=3))
    assert first["candidates"] == second["candidates"]
    assert first["model_name"] == second["model_name"]


def test_report_lists_every_model_and_the_winner(model_dir):
    meta = read_metadata(model_dir)
    text = format_report(meta)
    for candidate in meta["candidates"]:
        assert candidate["name"] in text
    assert f"Selected: {meta['model_name']}" in text
    assert "Saved as version v1" in text


def test_progress_messages_are_reported(tmp_path, dataset_csv):
    messages = []
    run(TrainSettings(data_path=dataset_csv, model_dir=tmp_path, cv_folds=3), progress=messages.append)
    assert any("Cross-validating" in text for text in messages)
    assert messages[-1].startswith("Saved v1")


def test_bad_data_stops_before_anything_is_saved(tmp_path):
    settings = TrainSettings(data_path=tmp_path / "missing.csv", model_dir=tmp_path / "models")
    with pytest.raises(DataError):
        run(settings)
    assert not (tmp_path / "models").exists()
