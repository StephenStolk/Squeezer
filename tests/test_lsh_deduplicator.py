from contextsqueezer.compressors.lsh_deduplicator import (
    deduplicate_turns,
    simhash,
    hamming_similarity,
    rabin_fingerprint,
)


LONG_DOC = (
    "This is a long documentation block that gets repeated across multiple "
    "conversation turns because the agent keeps re-reading the same file " * 3
)


def test_simhash_identical_text():
    a = simhash(LONG_DOC)
    b = simhash(LONG_DOC)
    assert a == b


def test_hamming_similarity_bounds():
    a = simhash(LONG_DOC)
    b = simhash(LONG_DOC + " extra")
    sim = hamming_similarity(a, b)
    assert 0.0 <= sim <= 1.0


def test_rabin_fingerprint_exact_match():
    assert rabin_fingerprint("hello world") == rabin_fingerprint("hello world")
    assert rabin_fingerprint("hello world") != rabin_fingerprint("hello worlds")


def test_deduplicate_turns_marks_exact_duplicate():
    messages = [
        {"role": "user", "content": LONG_DOC},
        {"role": "assistant", "content": "Got it, thanks."},
        {"role": "user", "content": LONG_DOC},  # exact duplicate
    ]
    out, saved = deduplicate_turns(messages, similarity_threshold=0.85)
    assert "[DEDUP:" in out[2]["content"]
    assert saved > 0


def test_deduplicate_turns_short_text_untouched():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "hi"},
    ]
    out, saved = deduplicate_turns(messages)
    assert out[1]["content"] == "hi"  # too short to dedupe
