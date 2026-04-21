import json


def parse_json_output(value):
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def parse_request_payload(request: str) -> dict:
    try:
        parsed = json.loads(request)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}
