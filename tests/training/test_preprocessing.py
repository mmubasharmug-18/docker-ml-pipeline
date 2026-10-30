import numpy as np
import pytest

from tests.helpers import make_dataset
from training.data import FEATURES, RENAMES, ZERO_AS_MISSING
from training.preprocessing import build_preprocessor


def training_frame():
    return make_dataset().rename(columns=RENAMES)[FEATURES]


@pytest.mark.parametrize("column", ZERO_AS_MISSING)
def test_a_zero_is_replaced_by_the_median_of_the_real_values(column):
    frame = training_frame()
    preprocessor = build_preprocessor().fit(frame)
    median = frame.loc[frame[column] != 0, column].median()

    with_zero = frame.iloc[[0]].copy()
    with_zero[column] = 0
    with_median = with_zero.copy()
    with_median[column] = median

    assert np.allclose(preprocessor.transform(with_zero), preprocessor.transform(with_median))


def test_zero_pregnancies_is_a_real_value_not_a_missing_one():
    frame = training_frame()
    preprocessor = build_preprocessor().fit(frame)
    none = frame.iloc[[0]].copy()
    none["pregnancies"] = 0
    some = none.copy()
    some["pregnancies"] = 1
    assert not np.allclose(preprocessor.transform(none), preprocessor.transform(some))


def test_output_is_scaled_to_mean_zero_and_unit_spread():
    frame = training_frame()
    ready = build_preprocessor().fit_transform(frame)
    assert ready.shape == (len(frame), len(FEATURES))
    assert np.allclose(ready.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(ready.std(axis=0), 1, atol=1e-6)


def test_output_has_no_missing_values():
    frame = training_frame()
    assert not np.isnan(build_preprocessor().fit_transform(frame)).any()


def test_two_fits_give_the_same_result():
    frame = training_frame()
    first = build_preprocessor().fit_transform(frame)
    second = build_preprocessor().fit_transform(frame)
    assert np.array_equal(first, second)
