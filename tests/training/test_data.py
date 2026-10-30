import pytest

from tests.helpers import make_dataset
from training.data import FEATURES, MIN_ROWS, TARGET, DataError, file_hash, load_dataset


def write(tmp_path, frame, name="diabetes.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_columns_are_renamed_and_ordered(tmp_path):
    frame = load_dataset(write(tmp_path, make_dataset()))
    assert list(frame.columns) == FEATURES + [TARGET]
    assert len(frame) == 300


def test_extra_columns_are_dropped(tmp_path):
    raw = make_dataset()
    raw["Id"] = range(len(raw))
    frame = load_dataset(write(tmp_path, raw))
    assert "Id" not in frame.columns


def test_zeros_are_left_alone_for_the_preprocessor(tmp_path):
    frame = load_dataset(write(tmp_path, make_dataset()))
    assert (frame["insulin"] == 0).any()


def test_missing_file_says_where_to_put_it(tmp_path):
    with pytest.raises(DataError) as error:
        load_dataset(tmp_path / "nope.csv")
    assert "Dataset not found" in str(error.value)


def test_missing_column_is_named(tmp_path):
    raw = make_dataset().drop(columns=["Glucose", "BMI"])
    with pytest.raises(DataError) as error:
        load_dataset(write(tmp_path, raw))
    assert "glucose" in str(error.value)
    assert "bmi" in str(error.value)


def test_text_in_a_number_column_is_refused(tmp_path):
    raw = make_dataset().astype(object)
    raw.loc[3, "Glucose"] = "abc"
    with pytest.raises(DataError) as error:
        load_dataset(write(tmp_path, raw))
    assert "glucose" in str(error.value)


def test_empty_cells_are_refused(tmp_path):
    raw = make_dataset().astype(float)
    raw.loc[5, "Age"] = None
    with pytest.raises(DataError) as error:
        load_dataset(write(tmp_path, raw))
    assert "age" in str(error.value)


def test_outcome_must_be_zero_or_one(tmp_path):
    raw = make_dataset()
    raw.loc[0, "Outcome"] = 2
    with pytest.raises(DataError) as error:
        load_dataset(write(tmp_path, raw))
    assert "0 and 1" in str(error.value)


def test_both_classes_are_needed(tmp_path):
    raw = make_dataset()
    raw["Outcome"] = 0
    with pytest.raises(DataError) as error:
        load_dataset(write(tmp_path, raw))
    assert "both" in str(error.value)


def test_too_few_rows_are_refused(tmp_path):
    raw = make_dataset(rows=MIN_ROWS - 1)
    raw.loc[0, "Outcome"] = 0
    raw.loc[1, "Outcome"] = 1
    with pytest.raises(DataError) as error:
        load_dataset(write(tmp_path, raw))
    assert "rows" in str(error.value)


def test_file_that_is_not_a_csv_gives_a_clear_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DataError):
        load_dataset(path)


def test_hash_follows_the_content(tmp_path):
    first = write(tmp_path, make_dataset(seed=1), "a.csv")
    same = write(tmp_path, make_dataset(seed=1), "b.csv")
    other = write(tmp_path, make_dataset(seed=2), "c.csv")
    assert file_hash(first) == file_hash(same)
    assert file_hash(first) != file_hash(other)
