import pytest
from pathlib import Path

from contextsqueezer.storage.sqlite_store import Store, init_db
from contextsqueezer.storage.ccr import CCRManager, estimate_tokens, make_pointer


@pytest.fixture
async def store(tmp_path: Path):
    db_path = tmp_path / "ccr_test.db"
    await init_db(db_path)
    async with Store(db_path) as s:
        yield s


async def test_ccr_offload_and_resolve(store):
    ccr = CCRManager(store, token_threshold=10)
    big_text = "hello world " * 50
    pointer = await ccr.maybe_offload(big_text, label="test")
    assert pointer != big_text
    assert "[CCR:" in pointer
    assert ccr.was_used is True

    resolved = await ccr.resolve_pointer(pointer)
    assert resolved == big_text


async def test_ccr_small_text_not_offloaded(store):
    ccr = CCRManager(store, token_threshold=2000)
    small_text = "short text"
    result = await ccr.maybe_offload(small_text)
    assert result == small_text
    assert ccr.was_used is False


async def test_ccr_tool_call_resolution(store):
    ccr = CCRManager(store, token_threshold=10)
    content = "important data " * 30
    pointer = await ccr.maybe_offload(content)
    hash_id = pointer.split(":")[1].split(" ")[0]
    result = await ccr.handle_tool_call({"hash": hash_id})
    assert result == content


async def test_ccr_unknown_hash_returns_error(store):
    ccr = CCRManager(store)
    result = await ccr.handle_tool_call({"hash": "deadbeef00000000"})
    assert "ERROR" in result


def test_estimate_tokens_positive():
    assert estimate_tokens("hello") > 0


def test_make_pointer_format():
    p = make_pointer("abc123", "label", 100)
    assert p == "[CCR:abc123 | label | ~100 tok]"
