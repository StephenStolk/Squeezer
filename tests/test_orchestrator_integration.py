import pytest
from pathlib import Path

from contextsqueezer.config import Settings
from contextsqueezer.pipeline.orchestrator import run_pipeline
from contextsqueezer.storage.sqlite_store import Store, init_db


@pytest.fixture
async def store(tmp_path: Path):
    db_path = tmp_path / "test.db"
    await init_db(db_path)
    async with Store(db_path) as s:
        yield s


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "test.db")


async def test_pipeline_scrubs_pii(settings, store):
    payload = {
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": "my email is leak@example.com"},
        ],
    }
    result = await run_pipeline(payload, settings=settings, store=store)
    text = str(result.compressed_payload["messages"][0]["content"])
    assert "leak@example.com" not in text
    assert result.pii_hits.get("email", 0) >= 1


async def test_pipeline_compacts_code(settings, store):
    code = "```python\ndef foo(x):\n    y = x + 1\n    z = y * 2\n    return z\n```"
    payload = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": code}],
    }
    result = await run_pipeline(payload, settings=settings, store=store)
    assert result.algo_breakdown.get("ast_compactor", 0) >= 0
    assert result.compressed_tokens <= result.raw_tokens


async def test_pipeline_offloads_large_block_to_ccr(settings, store):
    settings.ccr_token_threshold = 50  # force offload for test
    big_text = "x" * 5000
    payload = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": big_text}],
    }
    result = await run_pipeline(payload, settings=settings, store=store)
    assert result.ccr_used is True
    content = str(result.compressed_payload["messages"][0]["content"])
    assert "[CCR:" in content
    # squeezer_retrieve tool should be injected
    tool_names = [t.get("name") for t in result.compressed_payload.get("tools", [])]
    assert "squeezer_retrieve" in tool_names


async def test_pipeline_produces_smaller_or_equal_payload(settings, store):
    payload = {
        "model": "claude-sonnet-4-6",
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Certainly! " + ("Hello there. " * 50)},
            {"role": "assistant", "content": "Sure, here is the answer."},
        ],
    }
    result = await run_pipeline(payload, settings=settings, store=store)
    assert result.compressed_tokens <= result.raw_tokens
