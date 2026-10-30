from fastapi.testclient import TestClient

from serving.main import create_app


def make_client(settings):
    return TestClient(create_app(settings))
