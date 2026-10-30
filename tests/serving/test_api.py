import json

import pytest

from serving.config import Settings
from serving.schemas import FIELD_NAMES
from tests.helpers import HIGH_RISK_PAYLOAD, VALID_PAYLOAD
from tests.serving.clients import make_client


def predict(client, **changes):
    return client.post("/predict", json={**VALID_PAYLOAD, **changes})


def test_health_reports_the_model_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "v1"}


def test_model_endpoint_describes_the_serving_model(client):
    body = client.get("/model").json()
    assert body["version"] == "v1"
    assert body["current_version"] == "v1"
    assert body["available_versions"] == ["v1"]
    assert body["features"] == FIELD_NAMES
    assert body["selection_metric"] == "roc_auc"
    assert set(body["test_metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert body["sklearn_matches"] is True
    assert body["pinned_version"] is None


def test_predict_returns_a_complete_answer(client):
    response = predict(client)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "label", "probability", "threshold", "model_version", "model_name"}
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability"] <= 1
    assert body["model_version"] == "v1"
    assert body["label"] == ("diabetic" if body["prediction"] == 1 else "not diabetic")


def test_prediction_follows_the_probability_and_threshold(client):
    for payload in (VALID_PAYLOAD, HIGH_RISK_PAYLOAD):
        body = client.post("/predict", json=payload).json()
        assert body["prediction"] == int(body["probability"] >= body["threshold"])


def test_high_risk_patient_scores_higher_than_a_typical_one(client):
    typical = client.post("/predict", json=VALID_PAYLOAD).json()
    high = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()
    assert high["probability"] > typical["probability"]
    assert high["prediction"] == 1


def test_same_input_gives_the_same_answer(client):
    assert predict(client).json() == predict(client).json()


def test_whole_numbers_and_decimals_are_the_same_value(client):
    assert predict(client, glucose=120).json() == predict(client, glucose=120.0).json()


def test_zero_means_not_measured_and_is_accepted(client):
    response = predict(client, insulin=0, skin_thickness=0, glucose=0, blood_pressure=0, bmi=0)
    assert response.status_code == 200


@pytest.mark.parametrize(
    "field,value",
    [
        ("pregnancies", 0),
        ("pregnancies", 20),
        ("glucose", 0),
        ("glucose", 300),
        ("blood_pressure", 200),
        ("skin_thickness", 100),
        ("insulin", 900),
        ("bmi", 70),
        ("diabetes_pedigree", 0),
        ("diabetes_pedigree", 3),
        ("age", 1),
        ("age", 120),
    ],
)
def test_values_exactly_on_the_limits_are_accepted(client, field, value):
    assert predict(client, **{field: value}).status_code == 200


@pytest.mark.parametrize(
    "field,value",
    [
        ("pregnancies", -1),
        ("pregnancies", 21),
        ("glucose", -1),
        ("glucose", 301),
        ("blood_pressure", 201),
        ("skin_thickness", 101),
        ("insulin", 901),
        ("bmi", 70.1),
        ("diabetes_pedigree", -0.1),
        ("diabetes_pedigree", 3.1),
        ("age", 0),
        ("age", 121),
    ],
)
def test_values_outside_the_limits_are_rejected_with_400(client, field, value):
    response = predict(client, **{field: value})
    assert response.status_code == 400
    assert response.json()["detail"][0]["field"] == field


@pytest.mark.parametrize(
    "field,value",
    [
        ("glucose", "120"),
        ("glucose", None),
        ("glucose", True),
        ("glucose", [120]),
        ("bmi", "high"),
        ("age", 35.5),
        ("age", "35"),
        ("pregnancies", 2.0),
        ("pregnancies", True),
    ],
)
def test_wrong_types_are_rejected_with_400(client, field, value):
    response = predict(client, **{field: value})
    assert response.status_code == 400
    assert response.json()["detail"][0]["field"] == field


@pytest.mark.parametrize("field", FIELD_NAMES)
def test_every_field_is_required(client, field):
    payload = {key: value for key, value in VALID_PAYLOAD.items() if key != field}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == [{"field": field, "message": f"{field} is required"}]


def test_unknown_field_is_rejected(client):
    response = predict(client, colour="red")
    assert response.status_code == 400
    assert response.json()["detail"][0]["field"] == "colour"


def test_all_problems_are_listed_together(client):
    response = predict(client, age=150, glucose="high", bmi=-3)
    fields = [item["field"] for item in response.json()["detail"]]
    assert response.status_code == 400
    assert sorted(fields) == ["age", "bmi", "glucose"]


def test_nan_is_rejected(client):
    body = json.dumps({**VALID_PAYLOAD, "glucose": float("nan")})
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 400


@pytest.mark.parametrize("body", ["not json", "", "[]", "42", '"text"'])
def test_a_body_that_is_not_a_json_object_is_rejected(client, body):
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["detail"][0]["field"] == "body"


def test_a_bad_request_does_not_break_the_next_one(client):
    assert predict(client, age=999).status_code == 400
    assert predict(client).status_code == 200


def test_predict_only_accepts_post(client):
    assert client.get("/predict").status_code == 405


def test_health_without_a_model_is_503(empty_client):
    response = empty_client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "no_model"
    assert "No trained model" in response.json()["detail"]


def test_predict_without_a_model_is_503(empty_client):
    response = empty_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503
    assert "No trained model" in response.json()["detail"]


def test_model_endpoint_without_a_model_is_503(empty_client):
    assert empty_client.get("/model").status_code == 503


def test_input_is_still_checked_without_a_model(empty_client):
    assert empty_client.post("/predict", json={"age": 1}).status_code == 400


def test_newest_version_is_served_by_default(two_version_dir):
    with make_client(Settings(model_dir=two_version_dir)) as client:
        assert predict(client).json()["model_version"] == "v2"
        info = client.get("/model").json()
        assert info["version"] == "v2"
        assert info["available_versions"] == ["v1", "v2"]


def test_an_older_version_can_be_pinned(two_version_dir):
    with make_client(Settings(model_dir=two_version_dir, model_version="v1")) as client:
        assert predict(client).json()["model_version"] == "v1"
        info = client.get("/model").json()
        assert info["version"] == "v1"
        assert info["pinned_version"] == "v1"
        assert info["current_version"] == "v2"


def test_a_pinned_version_that_does_not_exist_serves_nothing(model_dir):
    with make_client(Settings(model_dir=model_dir, model_version="v99")) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert "v99" in response.json()["detail"]
        assert predict(client).status_code == 503
