import shutil

import pytest

from tests.helpers import make_dataset, train_into


@pytest.fixture(scope="session")
def dataset_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "diabetes.csv"
    make_dataset().to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, dataset_csv):
    folder = tmp_path_factory.mktemp("models")
    train_into(folder, dataset_csv)
    return folder


@pytest.fixture(scope="session")
def two_version_dir(tmp_path_factory, dataset_csv, model_dir):
    folder = tmp_path_factory.mktemp("models_two") / "models"
    shutil.copytree(model_dir, folder)
    train_into(folder, dataset_csv, random_state=7)
    return folder
