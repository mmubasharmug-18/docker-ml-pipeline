import json
import subprocess
import sys

from tests.helpers import VALID_PAYLOAD
from tests.server import ROOT, free_port, running_server, start_env


def run_module(module, **settings):
    return subprocess.run(
        [sys.executable, "-m", module],
        env=start_env(**settings),
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_server_starts_loads_the_model_and_predicts(model_dir):
    with running_server(MODEL_DIR=model_dir) as server:
        status, body = server.request("/health")
        assert status == 200
        assert json.loads(body)["model_version"] == "v1"

        status, body = server.request("/predict", json.dumps(VALID_PAYLOAD).encode())
        assert status == 200
        assert json.loads(body)["model_version"] == "v1"

        status, body = server.request("/predict", json.dumps({**VALID_PAYLOAD, "age": 500}).encode())
        assert status == 400
    assert "Loaded model v1" in server.output


def test_model_version_can_be_pinned_with_an_environment_variable(two_version_dir):
    with running_server(MODEL_DIR=two_version_dir, MODEL_VERSION="v1") as server:
        status, body = server.request("/predict", json.dumps(VALID_PAYLOAD).encode())
        assert json.loads(body)["model_version"] == "v1"
    assert "Loaded model v1" in server.output


def test_server_without_a_model_stays_up_but_is_not_ready(tmp_path):
    with running_server(MODEL_DIR=tmp_path / "nothing") as server:
        status, body = server.request("/health")
        assert status == 503
        assert json.loads(body)["status"] == "no_model"
    assert "No model loaded" in server.output


def test_bad_settings_stop_the_service_with_a_clear_message():
    done = run_module("serving", LOG_LEVEL="LOUD", APP_PORT=free_port())
    assert done.returncode == 2
    assert "LOG_LEVEL" in done.stderr


def test_healthcheck_passes_when_a_model_is_loaded(model_dir):
    with running_server(MODEL_DIR=model_dir) as server:
        done = run_module("serving.healthcheck", APP_PORT=server.port)
    assert done.returncode == 0


def test_healthcheck_fails_when_no_model_is_loaded(tmp_path):
    with running_server(MODEL_DIR=tmp_path / "nothing") as server:
        done = run_module("serving.healthcheck", APP_PORT=server.port)
    assert done.returncode == 1


def test_healthcheck_fails_when_nothing_is_listening():
    assert run_module("serving.healthcheck", APP_PORT=free_port()).returncode == 1
