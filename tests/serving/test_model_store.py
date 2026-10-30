import json
import shutil

import pytest

from serving.model_store import ModelNotAvailable, current_version, list_versions, load_model
from serving.schemas import FIELD_NAMES


def copy_models(model_dir, tmp_path):
    target = tmp_path / "models"
    shutil.copytree(model_dir, target)
    return target


def edit_metadata(folder, **changes):
    path = folder / "metadata.json"
    meta = json.loads(path.read_text())
    meta.update(changes)
    path.write_text(json.dumps(meta))


def test_loads_the_current_version(model_dir):
    loaded = load_model(model_dir)
    assert loaded.version == "v1"
    assert loaded.features == FIELD_NAMES
    assert loaded.threshold == 0.5
    assert loaded.sklearn_matches is True
    assert loaded.name in {"Logistic Regression", "Random Forest", "Gradient Boosting"}


def test_empty_folder_says_to_train_first(tmp_path):
    with pytest.raises(ModelNotAvailable) as error:
        load_model(tmp_path / "nothing")
    assert "No trained model" in str(error.value)


def test_unknown_version_lists_the_available_ones(model_dir):
    with pytest.raises(ModelNotAvailable) as error:
        load_model(model_dir, "v99")
    assert "v99" in str(error.value)
    assert "v1" in str(error.value)


def test_newest_version_is_the_default_and_an_old_one_can_be_asked_for(two_version_dir):
    assert current_version(two_version_dir) == "v2"
    assert list_versions(two_version_dir) == ["v1", "v2"]
    assert load_model(two_version_dir).version == "v2"
    assert load_model(two_version_dir, "v1").version == "v1"


def test_incomplete_version_is_refused(model_dir, tmp_path):
    models = copy_models(model_dir, tmp_path)
    (models / "v1" / "preprocessor.joblib").unlink()
    with pytest.raises(ModelNotAvailable) as error:
        load_model(models)
    assert "preprocessor.joblib" in str(error.value)


def test_corrupt_model_file_is_refused(model_dir, tmp_path):
    models = copy_models(model_dir, tmp_path)
    (models / "v1" / "model.joblib").write_bytes(b"this is not a model")
    with pytest.raises(ModelNotAvailable) as error:
        load_model(models)
    assert "Could not load" in str(error.value)


def test_model_trained_on_other_features_is_refused(model_dir, tmp_path):
    models = copy_models(model_dir, tmp_path)
    edit_metadata(models / "v1", features=["a", "b"])
    with pytest.raises(ModelNotAvailable) as error:
        load_model(models)
    assert "expects" in str(error.value)


def test_different_scikit_learn_version_is_flagged(model_dir, tmp_path):
    models = copy_models(model_dir, tmp_path)
    meta = json.loads((models / "v1" / "metadata.json").read_text())
    meta["libraries"]["scikit_learn"] = "0.0.1"
    edit_metadata(models / "v1", libraries=meta["libraries"])
    assert load_model(models).sklearn_matches is False
