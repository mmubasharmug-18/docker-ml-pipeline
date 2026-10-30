from pathlib import Path

import pytest

from serving.config import ConfigError, Settings, load_settings


def test_defaults_when_nothing_is_set():
    assert load_settings({}) == Settings()


def test_values_are_read_from_the_environment():
    env = {
        "APP_HOST": "0.0.0.0",
        "APP_PORT": "9000",
        "LOG_LEVEL": "debug",
        "MODEL_DIR": "/models",
        "MODEL_VERSION": "v3",
    }
    settings = load_settings(env)
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.log_level == "DEBUG"
    assert settings.model_dir == Path("/models")
    assert settings.model_version == "v3"


def test_empty_model_version_means_use_the_current_one():
    assert load_settings({"MODEL_VERSION": ""}).model_version is None


@pytest.mark.parametrize("version", ["3", "v", "v1.2", "latest", "V1"])
def test_bad_model_version_is_rejected(version):
    with pytest.raises(ConfigError) as error:
        load_settings({"MODEL_VERSION": version})
    assert "MODEL_VERSION" in str(error.value)


@pytest.mark.parametrize("level", ["LOUD", "5"])
def test_bad_log_level_is_rejected(level):
    with pytest.raises(ConfigError) as error:
        load_settings({"LOG_LEVEL": level})
    assert "LOG_LEVEL" in str(error.value)


@pytest.mark.parametrize("port", ["abc", "0", "65536", "-1"])
def test_bad_port_is_rejected(port):
    with pytest.raises(ConfigError) as error:
        load_settings({"APP_PORT": port})
    assert "APP_PORT" in str(error.value)
