from contextsqueezer.pipeline.classifier import classify_text, ContentKind, split_into_blocks


def test_classify_code():
    text = "```python\ndef foo():\n    return 1\n```"
    assert classify_text(text) == ContentKind.CODE


def test_classify_json():
    text = '{"key": "value", "nested": {"a": 1}}'
    assert classify_text(text) == ContentKind.JSON_DATA


def test_classify_shell():
    text = "Traceback (most recent call last):\n  File \"x.py\", line 1\nError: bad"
    assert classify_text(text) == ContentKind.SHELL_OUTPUT


def test_classify_conversation():
    text = "Hey, how's it going today? Just checking in."
    assert classify_text(text) == ContentKind.CONVERSATION


def test_split_into_blocks_extracts_code():
    text = "Here is some code:\n```python\ndef foo(): pass\n```\nThat's it."
    blocks = split_into_blocks(text)
    code_blocks = [b for b in blocks if b.kind == ContentKind.CODE]
    assert len(code_blocks) == 1
