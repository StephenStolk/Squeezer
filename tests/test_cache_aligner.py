from contextsqueezer.pipeline.cache_aligner import align_for_cache


def _conv(n: int) -> list[dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
    ] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
        for i in range(n)
    ]


def test_align_preserves_chronological_order():
    messages = _conv(10)
    out, _, _ = align_for_cache(messages, dynamic_tail_size=3)
    # Content order must be byte-identical to input — only cache_control added.
    for original, aligned in zip(messages, out):
        orig_content = original["content"]
        aligned_content = aligned["content"]
        if isinstance(aligned_content, list):
            aligned_text = aligned_content[0]["text"] if aligned_content else ""
        else:
            aligned_text = aligned_content
        assert orig_content == aligned_text


def test_growing_conversation_keeps_stable_prefix():
    """
    The defining correctness property: as a conversation grows by one turn,
    the prefix up to the *old* boundary must remain byte-identical — new
    content can only ever be appended, never inserted into the middle.
    """
    conv_a = _conv(10)
    conv_b = _conv(11)  # one more turn appended

    out_a, _, boundary_a = align_for_cache(conv_a, dynamic_tail_size=3)
    out_b, _, boundary_b = align_for_cache(conv_b, dynamic_tail_size=3)

    def _flatten(msg):
        c = msg["content"]
        if isinstance(c, list):
            return c[0]["text"] if c else ""
        return c

    prefix_a = [_flatten(m) for m in out_a[:boundary_a]]
    prefix_b = [_flatten(m) for m in out_b[:boundary_a]]  # same length as old boundary
    assert prefix_a == prefix_b


def test_tools_sorted_alphabetically():
    tools = [{"name": "zebra"}, {"name": "apple"}, {"name": "mango"}]
    _, sorted_tools, _ = align_for_cache([{"role": "user", "content": "hi"}], tools)
    assert [t["name"] for t in sorted_tools] == ["apple", "mango", "zebra"]


def test_empty_messages_handled():
    out, tools, boundary = align_for_cache([], [])
    assert out == []
    assert boundary == 0
