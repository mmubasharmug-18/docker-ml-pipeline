import json

import pytest

from training.registry import current_version, list_versions, next_version, save_version, set_current


def dummy(folder, version):
    return save_version(folder, version, {"model": 1}, {"preprocessor": 1}, {"version": version})


def test_first_version_is_v1(tmp_path):
    assert next_version(tmp_path / "models") == "v1"


def test_versions_are_sorted_by_number_not_by_text(tmp_path):
    for name in ("v1", "v2", "v10"):
        (tmp_path / name).mkdir()
    assert list_versions(tmp_path) == ["v1", "v2", "v10"]
    assert next_version(tmp_path) == "v11"


def test_other_folders_and_files_are_ignored(tmp_path):
    (tmp_path / "v1").mkdir()
    (tmp_path / "backup").mkdir()
    (tmp_path / "v2x").mkdir()
    (tmp_path / "v3").write_text("a file, not a folder")
    assert list_versions(tmp_path) == ["v1"]


def test_save_writes_the_three_artifacts_and_no_temp_folder(tmp_path):
    folder = dummy(tmp_path, "v1")
    assert sorted(path.name for path in folder.iterdir()) == ["metadata.json", "model.joblib", "preprocessor.joblib"]
    assert [path.name for path in tmp_path.iterdir()] == ["v1"]
    assert json.loads((folder / "metadata.json").read_text())["version"] == "v1"


def test_a_saved_version_is_never_overwritten(tmp_path):
    dummy(tmp_path, "v1")
    with pytest.raises(FileExistsError):
        dummy(tmp_path, "v1")


def test_current_version_follows_set_current(tmp_path):
    assert current_version(tmp_path) is None
    set_current(tmp_path, "v1")
    assert current_version(tmp_path) == "v1"
    set_current(tmp_path, "v2")
    assert current_version(tmp_path) == "v2"
    assert [path.name for path in tmp_path.iterdir()] == ["current.json"]


def test_broken_current_file_means_no_current_version(tmp_path):
    (tmp_path / "current.json").write_text("not json")
    assert current_version(tmp_path) is None
