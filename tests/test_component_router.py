import pytest
from pathlib import Path

from contextsqueezer.storage.sqlite_store import Store, init_db
from contextsqueezer.storage.ccr import CCRManager
from contextsqueezer.pipeline.component_router import (
    ComponentRouter,
    SqueezerMeta,
    extract_meta,
)


@pytest.fixture
async def store(tmp_path: Path):
    db_path = tmp_path / "comp_test.db"
    await init_db(db_path)
    async with Store(db_path) as s:
        yield s


LONG_CHUNK = "This is a retrieved document chunk about Junos OS BGP session flap handling. " * 5


def test_extract_meta_from_headers():
    payload = {"messages": []}
    headers = {
        "X-Squeezer-Component": "retriever",
        "X-Squeezer-Run": "run-1",
        "X-Squeezer-Budget": "5000",
    }
    meta = extract_meta(payload, headers)
    assert meta.component_id == "retriever"
    assert meta.run_id == "run-1"
    assert meta.budget_tokens == 5000


def test_extract_meta_from_body_and_strips_it():
    payload = {
        "messages": [],
        "squeezer_meta": {"component_id": "planner", "run_id": "run-2", "budget_tokens": 3000},
    }
    meta = extract_meta(payload, {})
    assert meta.component_id == "planner"
    assert "squeezer_meta" not in payload  # popped out before forwarding upstream


def test_extract_meta_headers_take_precedence():
    payload = {"messages": [], "squeezer_meta": {"component_id": "from_body"}}
    headers = {"X-Squeezer-Component": "from_header"}
    meta = extract_meta(payload, headers)
    assert meta.component_id == "from_header"


def test_meta_disabled_without_both_fields():
    meta = SqueezerMeta(component_id="solo", run_id=None)
    assert meta.cross_component_enabled is False


async def test_cross_component_dedup_replaces_second_occurrence(store):
    ccr = CCRManager(store)
    meta = SqueezerMeta(component_id="agent_a", run_id="run-x")
    router_a = ComponentRouter(store, meta, ccr=ccr)
    text_a, saved_a = await router_a.dedupe_against_run(LONG_CHUNK)
    assert text_a == LONG_CHUNK  # first time seen — untouched
    assert saved_a == 0

    meta_b = SqueezerMeta(component_id="agent_b", run_id="run-x")
    router_b = ComponentRouter(store, meta_b, ccr=ccr)
    text_b, saved_b = await router_b.dedupe_against_run(LONG_CHUNK)
    assert "[CCR:" in text_b
    assert saved_b > 0


async def test_cross_component_pointer_is_actually_retrievable(store):
    """The whole point of routing through CCR: the pointer isn't a dead end."""
    ccr = CCRManager(store)
    meta_a = SqueezerMeta(component_id="agent_a", run_id="run-z")
    await ComponentRouter(store, meta_a, ccr=ccr).dedupe_against_run(LONG_CHUNK)

    meta_b = SqueezerMeta(component_id="agent_b", run_id="run-z")
    router_b = ComponentRouter(store, meta_b, ccr=ccr)
    pointer, _ = await router_b.dedupe_against_run(LONG_CHUNK)

    resolved = await ccr.resolve_pointer(pointer)
    assert resolved == LONG_CHUNK
    assert ccr.was_used is True  # squeezer_retrieve tool injection gets triggered


async def test_dedup_without_ccr_skips_rather_than_emits_dead_pointer(store):
    """No CCRManager supplied -> fail safe (skip), never emit an unresolvable pointer."""
    meta_a = SqueezerMeta(component_id="agent_a", run_id="run-w")
    await ComponentRouter(store, meta_a, ccr=None).dedupe_against_run(LONG_CHUNK)

    meta_b = SqueezerMeta(component_id="agent_b", run_id="run-w")
    text, saved = await ComponentRouter(store, meta_b, ccr=None).dedupe_against_run(LONG_CHUNK)
    assert text == LONG_CHUNK  # not replaced with an unresolvable pointer
    assert saved == 0


async def test_same_component_does_not_dedupe_against_itself(store):
    ccr = CCRManager(store)
    meta = SqueezerMeta(component_id="agent_a", run_id="run-y")
    router = ComponentRouter(store, meta, ccr=ccr)
    await router.dedupe_against_run(LONG_CHUNK)
    text, saved = await router.dedupe_against_run(LONG_CHUNK)
    # Same component re-sending its own content isn't cross-component redundancy
    assert text == LONG_CHUNK
    assert saved == 0


async def test_different_runs_do_not_share_ledger(store):
    ccr = CCRManager(store)
    meta_a = SqueezerMeta(component_id="agent_a", run_id="run-1")
    await ComponentRouter(store, meta_a, ccr=ccr).dedupe_against_run(LONG_CHUNK)

    meta_b = SqueezerMeta(component_id="agent_b", run_id="run-2")  # different run
    text, saved = await ComponentRouter(store, meta_b, ccr=ccr).dedupe_against_run(LONG_CHUNK)
    assert text == LONG_CHUNK  # no cross-run leakage
    assert saved == 0


async def test_disabled_router_is_noop(store):
    meta = SqueezerMeta(component_id=None, run_id=None)
    router = ComponentRouter(store, meta)
    assert router.enabled is False
    text, saved = await router.dedupe_against_run(LONG_CHUNK)
    assert text == LONG_CHUNK
    assert saved == 0
