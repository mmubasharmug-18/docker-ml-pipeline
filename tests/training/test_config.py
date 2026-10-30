from pathlib import Path

import pytest

from training.config import ConfigError, TrainSettings, load_settings


def test_defaults_when_nothing_is_set():
    assert load_settings({}) == TrainSettings()


def test_values_are_read_from_the_environment():
    env = {
        "DATA_PATH": "/data/diabetes.csv",
        "MODEL_DIR": "/models",
        "SELECTION_METRIC": "f1",
        "TEST_SIZE": "0.25",
        "RANDOM_STATE": "7",
        "CV_FOLDS": "3",
    }
    settings = load_settings(env)
    assert settings.data_path == Path("/data/diabetes.csv")
    assert settings.model_dir == Path("/models")
    assert settings.selection_metric == "f1"
    assert settings.test_size == 0.25
    assert settings.random_state == 7
    assert settings.cv_folds == 3


def test_metric_is_not_case_sensitive():
    assert load_settings({"SELECTION_METRIC": "ROC_AUC"}).selection_metric == "roc_auc"


def test_empty_values_fall_back_to_defaults():
    env = {name: "" for name in ("DATA_PATH", "MODEL_DIR", "SELECTION_METRIC", "TEST_SIZE", "RANDOM_STATE", "CV_FOLDS")}
    assert load_settings(env) == TrainSettings()


@pytest.mark.parametrize("metric", ["auc", "LOUD", "5", "roc auc"])
def test_unknown_metric_is_rejected(metric):
    with pytest.raises(ConfigError) as error:
        load_settings({"SELECTION_METRIC": metric})
    assert "SELECTION_METRIC" in str(error.value)


@pytest.mark.parametrize(
    "name,value",
    [
        ("TEST_SIZE", "0.9"),
        ("TEST_SIZE", "0.01"),
        ("TEST_SIZE", "abc"),
        ("CV_FOLDS", "1"),
        ("CV_FOLDS", "20"),
        ("CV_FOLDS", "2.5"),
        ("RANDOM_STATE", "x"),
    ],
)
def test_bad_numbers_are_rejected(name, value):
    with pytest.raises(ConfigError) as error:
        load_settings({name: value})
    assert name in str(error.value)


def test_every_problem_is_reported_together():
    with pytest.raises(ConfigError) as error:
        load_settings({"SELECTION_METRIC": "LOUD", "CV_FOLDS": "99"})
    message = str(error.value)
    assert "SELECTION_METRIC" in message
    assert "CV_FOLDS" in message
