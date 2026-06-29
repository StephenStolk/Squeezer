import pytest
from pathlib import Path

from contextsqueezer.storage.sqlite_store import Store, init_db


@pytest.fixture
async def store(tmp_path: Path):
    db_path = tmp_path / "dash_test.db"
    await init_db(db_path)
    async with Store(db_path) as s:
        yield s


async def test_dashboard_summary_empty_db(store):
    summary = await store.dashboard_summary()
    assert summary["total_requests"] == 0
    assert summary["component_breakdown"] == []


async def test_dashboard_summary_aggregates_basic_metrics(store):
    await store.record_metrics(
        request_id="r1", raw_tokens=1000, compressed_tokens=400,
        proxy_latency_ms=5.0, upstream_latency_ms=200.0,
        algo_breakdown={"ast_compactor": 600}, cache_hit=True, ccr_used=False,
    )
    summary = await store.dashboard_summary()
    assert summary["total_requests"] == 1
    assert summary["total_tokens_saved"] == 600
    assert summary["cache_hits"] == 1
    assert summary["algo_breakdown"]["ast_compactor"] == 600


async def test_dashboard_summary_per_component_breakdown(store):
    await store.record_metrics(
        request_id="r1", raw_tokens=1000, compressed_tokens=900,
        proxy_latency_ms=1.0, upstream_latency_ms=1.0, algo_breakdown={},
        component_id="retriever_agent", run_id="run-1",
    )
    await store.record_metrics(
        request_id="r2", raw_tokens=500, compressed_tokens=50,
        proxy_latency_ms=1.0, upstream_latency_ms=1.0, algo_breakdown={},
        component_id="planner_agent", run_id="run-1",
    )
    await store.record_metrics(
        request_id="r3", raw_tokens=2000, compressed_tokens=2000,
        proxy_latency_ms=1.0, upstream_latency_ms=1.0, algo_breakdown={},
        component_id="",  # no component tagging — shouldn't appear in breakdown
    )

    summary = await store.dashboard_summary()
    breakdown = {c["component_id"]: c for c in summary["component_breakdown"]}

    assert "retriever_agent" in breakdown
    assert "planner_agent" in breakdown
    assert "" not in breakdown  # untagged requests excluded
    assert breakdown["planner_agent"]["tokens_saved"] == 450
    # Sorted by tokens_saved descending — planner_agent (450 saved) beats
    # retriever_agent (100 saved).
    ids_in_order = [c["component_id"] for c in summary["component_breakdown"]]
    assert ids_in_order.index("planner_agent") < ids_in_order.index("retriever_agent")
