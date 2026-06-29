import json

from contextsqueezer.compressors.json_crusher import crush_json, crush_json_in_text


def test_crush_json_basic_depth_clamp():
    data = {"a": {"b": {"c": {"d": {"e": "too deep"}}}}}
    compact, saved = crush_json(data, max_depth=2, use_kneedle=False)
    parsed = json.loads(compact)
    assert "a" in parsed


def test_crush_json_array_truncation():
    data = {"items": list(range(100))}
    compact, saved = crush_json(data, max_depth=4, use_kneedle=False)
    parsed = json.loads(compact)
    assert len(parsed["items"]) < 100


def test_crush_json_invalid_returns_passthrough():
    bad_json = "{not valid json"
    result, saved = crush_json(bad_json)
    assert result == bad_json
    assert saved == 0


def test_crush_json_in_text_embeds():
    text = 'Here is some data: {"key": "value", "nested": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}} end.'
    result, saved = crush_json_in_text(text)
    assert isinstance(result, str)
