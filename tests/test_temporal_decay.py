from contextsqueezer.compressors.temporal_decay import apply_temporal_decay


def _make_messages(n: int) -> list:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"This is turn number {i} with some content about `file_{i}.py` and the function process_data."}
        for i in range(n)
    ]


def test_recent_turns_verbatim():
    messages = _make_messages(15)
    out, saved = apply_temporal_decay(messages, recent_turns=2, partial_turns=8)
    # Last 2 turns should be untouched
    assert out[-1]["content"] == messages[-1]["content"]
    assert out[-2]["content"] == messages[-2]["content"]


def test_old_turns_condensed():
    messages = _make_messages(15)
    out, saved = apply_temporal_decay(messages, recent_turns=2, partial_turns=8)
    # Oldest turn (index 0) should be condensed to keyword digest
    assert out[0]["content"].startswith("[")
    assert saved >= 0


def test_system_message_preserved():
    messages = [{"role": "system", "content": "System instructions here."}] + _make_messages(15)
    out, saved = apply_temporal_decay(messages, recent_turns=2, partial_turns=8)
    assert out[0]["content"] == "System instructions here."


def test_pinned_old_turn_survives_decay():
    messages = _make_messages(15)
    # Pin the oldest turn — it would normally be condensed to a keyword digest.
    messages[0]["content"] = "[PIN] " + messages[0]["content"]
    out, saved = apply_temporal_decay(messages, recent_turns=2, partial_turns=8)
    assert out[0]["content"] == _make_messages(15)[0]["content"]  # marker stripped, full text kept
    assert not out[0]["content"].startswith("[")


def test_unpinned_turn_at_same_age_still_decays():
    messages = _make_messages(15)
    out, saved = apply_temporal_decay(messages, recent_turns=2, partial_turns=8)
    assert out[0]["content"].startswith("[")  # condensed as usual, no pin marker present

