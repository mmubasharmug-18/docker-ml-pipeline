def short(value) -> str:
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "..."


def field_name(loc) -> str:
    return ".".join(str(part) for part in loc if part != "body")


def describe(error: dict) -> dict:
    kind = error.get("type")
    if kind == "json_invalid":
        return {"field": "body", "message": "request body must be valid JSON"}

    name = field_name(error.get("loc", ()))
    if name == "":
        return {"field": "body", "message": "request body must be a JSON object with all the fields"}

    if kind == "missing":
        message = f"{name} is required"
    elif kind == "extra_forbidden":
        message = f"{name} is not a known field"
    else:
        message = f"{name}: {error.get('msg')} (got {short(error.get('input'))})"
    return {"field": name, "message": message}


def format_errors(errors: list) -> list:
    seen = set()
    result = []
    for error in errors:
        item = describe(error)
        key = (item["field"], item["message"])
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
