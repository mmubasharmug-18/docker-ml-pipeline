from serving.errors import format_errors


def test_missing_field():
    result = format_errors([{"type": "missing", "loc": ("body", "glucose"), "msg": "Field required", "input": {}}])
    assert result == [{"field": "glucose", "message": "glucose is required"}]


def test_unknown_field():
    result = format_errors([{"type": "extra_forbidden", "loc": ("body", "colour"), "msg": "x", "input": "red"}])
    assert result == [{"field": "colour", "message": "colour is not a known field"}]


def test_value_problem_shows_the_bad_value():
    result = format_errors(
        [{"type": "less_than_equal", "loc": ("body", "age"), "msg": "Input should be less than or equal to 120", "input": 150}]
    )
    assert result[0]["field"] == "age"
    assert "less than or equal to 120" in result[0]["message"]
    assert "got 150" in result[0]["message"]


def test_broken_json():
    result = format_errors([{"type": "json_invalid", "loc": ("body", 0), "msg": "bad", "input": {}}])
    assert result == [{"field": "body", "message": "request body must be valid JSON"}]


def test_whole_body_of_the_wrong_kind():
    result = format_errors([{"type": "model_attributes_type", "loc": ("body",), "msg": "x", "input": []}])
    assert result[0]["field"] == "body"


def test_long_values_are_shortened():
    result = format_errors([{"type": "float_parsing", "loc": ("body", "bmi"), "msg": "bad", "input": "x" * 200}])
    assert "..." in result[0]["message"]
    assert len(result[0]["message"]) < 120


def test_duplicates_are_listed_once():
    error = {"type": "missing", "loc": ("body", "age"), "msg": "x", "input": {}}
    assert len(format_errors([error, error])) == 1
