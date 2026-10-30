import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tests.helpers import make_dataset
from training.data import FEATURES, RENAMES, TARGET
from training.evaluate import SCORES, confusion_counts, cross_validated_scores, holdout_scores, pick_best
from training.preprocessing import build_preprocessor


class FixedModel:
    def __init__(self, probabilities):
        self.probabilities = np.array(probabilities, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.probabilities, self.probabilities])


Y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
PROBABILITIES = [0.9, 0.8, 0.2, 0.7, 0.1, 0.2, 0.3, 0.4]
X = np.zeros((8, 1))


def test_scores_match_a_hand_calculation():
    scores = holdout_scores(FixedModel(PROBABILITIES), X, Y)
    assert scores["accuracy"] == 0.75
    assert scores["precision"] == 0.6667
    assert scores["recall"] == 0.6667
    assert scores["f1"] == 0.6667
    assert scores["roc_auc"] == 0.7667


def test_confusion_counts_match_a_hand_calculation():
    assert confusion_counts(FixedModel(PROBABILITIES), X, Y) == {"tn": 4, "fp": 1, "fn": 1, "tp": 2}


def test_no_positive_predictions_gives_zero_not_an_error():
    scores = holdout_scores(FixedModel([0.1] * 8), X, Y)
    assert scores["precision"] == 0.0
    assert scores["recall"] == 0.0
    assert scores["f1"] == 0.0


def test_a_lower_threshold_catches_more_positives():
    model = FixedModel(PROBABILITIES)
    strict = holdout_scores(model, X, Y, threshold=0.5)
    loose = holdout_scores(model, X, Y, threshold=0.15)
    assert loose["recall"] > strict["recall"]
    assert loose["precision"] < strict["precision"]


def test_best_model_has_the_highest_value_of_the_asked_metric():
    scores = {
        "A": {"roc_auc": 0.80, "recall": 0.50},
        "B": {"roc_auc": 0.85, "recall": 0.40},
        "C": {"roc_auc": 0.70, "recall": 0.90},
    }
    assert pick_best(scores, "roc_auc") == "B"
    assert pick_best(scores, "recall") == "C"


def test_first_model_wins_a_tie():
    scores = {"A": {"f1": 0.7}, "B": {"f1": 0.7}}
    assert pick_best(scores, "f1") == "A"


def test_cross_validation_returns_every_metric_between_zero_and_one():
    frame = make_dataset().rename(columns=RENAMES)
    full = Pipeline([("preprocess", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))])
    scores = cross_validated_scores(full, frame[FEATURES], frame[TARGET], folds=3, random_state=0)
    assert set(scores) == set(SCORES)
    assert all(0 <= value <= 1 for value in scores.values())
    assert scores["roc_auc"] > 0.7
