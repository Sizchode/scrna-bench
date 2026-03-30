from __future__ import annotations

import json


def extract_and_parse_json(response: str):
    try:
        start = response.index("{")
        end = response.rindex("}")
        return json.loads(response[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        return None
