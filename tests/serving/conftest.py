import pytest

from serving.config import Settings
from tests.serving.clients import make_client


@pytest.fixture
def client(model_dir):
    with make_client(Settings(model_dir=model_dir)) as test_client:
        yield test_client


@pytest.fixture
def empty_client(tmp_path):
    with make_client(Settings(model_dir=tmp_path / "nothing")) as test_client:
        yield test_client
