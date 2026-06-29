"""
SQLite WAL-mode storage layer.

Tables
------
ccr_store   – Content-Compressed Retrieval: stores aggressively-stripped
              raw chunks that the upstream LLM can pull back on demand.
metrics     – Per-request token accounting, latency telemetry, and
              per-algorithm savings breakdown for the dashboard.
pii_log     – Local-only audit log of PII interceptions.
cache_hits  – Tracks upstream provider cache-hit signals.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import aiosqlite

_CREATE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS ccr_store (
        hash        TEXT PRIMARY KEY,
        content     TEXT    NOT NULL,
        label       TEXT    NOT NULL DEFAULT '',
        size_bytes  INTEGER NOT NULL DEFAULT 0,
        created_at  REAL    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS metrics (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id          TEXT    NOT NULL,
        ts                  REAL    NOT NULL,
        raw_tokens          INTEGER NOT NULL DEFAULT 0,
        compressed_tokens   INTEGER NOT NULL DEFAULT 0,
        proxy_latency_ms    REAL    NOT NULL DEFAULT 0,
        upstream_latency_ms REAL    NOT NULL DEFAULT 0,
        algo_breakdown      TEXT    NOT NULL DEFAULT '{}',
        cache_hit           INTEGER NOT NULL DEFAULT 0,
        ccr_used            INTEGER NOT NULL DEFAULT 0,
        component_id        TEXT    NOT NULL DEFAULT '',
        run_id               TEXT    NOT NULL DEFAULT '',
        budget_tier          INTEGER NOT NULL DEFAULT -1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pii_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          REAL    NOT NULL,
        pattern     TEXT    NOT NULL,
        count       INTEGER NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cache_hits (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          REAL    NOT NULL,
        model       TEXT    NOT NULL,
        hit         INTEGER NOT NULL DEFAULT 0,
        input_tokens INTEGER NOT NULL DEFAULT 0,
        cache_tokens INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS file_versions (
        file_path    TEXT    NOT NULL,
        version      INTEGER NOT NULL,
        content_hash TEXT    NOT NULL,
        content      TEXT    NOT NULL,
        created_at   REAL    NOT NULL,
        PRIMARY KEY (file_path, version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS xcomp_ledger (
        run_id       TEXT    NOT NULL,
        content_hash TEXT    NOT NULL,
        component_id TEXT    NOT NULL,
        created_at   REAL    NOT NULL,
        PRIMARY KEY (run_id, content_hash)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_metrics_ts  ON metrics(ts)",
    "CREATE INDEX IF NOT EXISTS idx_metrics_component ON metrics(component_id)",
    "CREATE INDEX IF NOT EXISTS idx_ccr_created ON ccr_store(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_file_versions_path ON file_versions(file_path)",
    "CREATE INDEX IF NOT EXISTS idx_xcomp_run ON xcomp_ledger(run_id)",
]


async def init_db(db_path: Path) -> None:
    """Initialise tables and enable WAL mode."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute("PRAGMA temp_store=MEMORY")
        for stmt in _CREATE_STATEMENTS:
            await db.execute(stmt)
        await db.commit()


def content_hash(data: str) -> str:
    """SHA-256 fingerprint for a text chunk (hex, 16 chars)."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]


class Store:
    """Async context-manager wrapper around aiosqlite."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._db: aiosqlite.Connection | None = None

    async def __aenter__(self) -> "Store":
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._db:
            await self._db.close()

    # ── CCR ──────────────────────────────────────────────────────────────────

    async def ccr_put(self, content: str, label: str = "") -> str:
        """Store a content chunk; return its hash ID."""
        h = content_hash(content)
        assert self._db
        await self._db.execute(
            "INSERT OR REPLACE INTO ccr_store(hash, content, label, size_bytes, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (h, content, label, len(content.encode()), time.time()),
        )
        await self._db.commit()
        return h

    async def ccr_get(self, hash_id: str) -> str | None:
        """Fetch a content chunk by hash ID."""
        assert self._db
        row = await (
            await self._db.execute(
                "SELECT content FROM ccr_store WHERE hash = ?", (hash_id,)
            )
        ).fetchone()
        return row["content"] if row else None

    async def ccr_count(self) -> int:
        assert self._db
        row = await (
            await self._db.execute("SELECT COUNT(*) as n FROM ccr_store")
        ).fetchone()
        return row["n"]

    # ── Metrics ───────────────────────────────────────────────────────────────

    async def record_metrics(
        self,
        *,
        request_id: str,
        raw_tokens: int,
        compressed_tokens: int,
        proxy_latency_ms: float,
        upstream_latency_ms: float,
        algo_breakdown: dict[str, int],
        cache_hit: bool = False,
        ccr_used: bool = False,
        component_id: str = "",
        run_id: str = "",
        budget_tier: int = -1,
    ) -> None:
        assert self._db
        await self._db.execute(
            "INSERT INTO metrics(request_id, ts, raw_tokens, compressed_tokens, "
            "proxy_latency_ms, upstream_latency_ms, algo_breakdown, cache_hit, ccr_used, "
            "component_id, run_id, budget_tier) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                request_id,
                time.time(),
                raw_tokens,
                compressed_tokens,
                proxy_latency_ms,
                upstream_latency_ms,
                json.dumps(algo_breakdown),
                int(cache_hit),
                int(ccr_used),
                component_id,
                run_id,
                budget_tier,
            ),
        )
        await self._db.commit()

    async def dashboard_summary(self) -> dict[str, Any]:
        """Aggregate stats for the dashboard API."""
        assert self._db

        row = await (
            await self._db.execute(
                "SELECT COUNT(*) as n, SUM(raw_tokens) as rt, "
                "SUM(compressed_tokens) as ct, AVG(proxy_latency_ms) as pl, "
                "AVG(upstream_latency_ms) as ul, SUM(cache_hit) as ch, "
                "SUM(ccr_used) as cu FROM metrics"
            )
        ).fetchone()

        recent_rows = await (
            await self._db.execute(
                "SELECT ts, raw_tokens, compressed_tokens FROM metrics "
                "ORDER BY ts DESC LIMIT 50"
            )
        ).fetchall()

        algo_totals: dict[str, int] = {}
        algo_rows = await (
            await self._db.execute("SELECT algo_breakdown FROM metrics")
        ).fetchall()
        for ar in algo_rows:
            try:
                bd = json.loads(ar["algo_breakdown"])
                for k, v in bd.items():
                    algo_totals[k] = algo_totals.get(k, 0) + v
            except (json.JSONDecodeError, TypeError):
                pass

        rt = row["rt"] or 0
        ct = row["ct"] or 0
        saved = rt - ct

        component_rows = await (
            await self._db.execute(
                "SELECT component_id, COUNT(*) as n, SUM(raw_tokens) as rt, "
                "SUM(compressed_tokens) as ct FROM metrics "
                "WHERE component_id != '' GROUP BY component_id "
                "ORDER BY (SUM(raw_tokens) - SUM(compressed_tokens)) DESC"
            )
        ).fetchall()
        component_breakdown = [
            {
                "component_id": r["component_id"],
                "requests": r["n"],
                "raw_tokens": r["rt"] or 0,
                "compressed_tokens": r["ct"] or 0,
                "tokens_saved": (r["rt"] or 0) - (r["ct"] or 0),
            }
            for r in component_rows
        ]

        return {
            "total_requests": row["n"] or 0,
            "total_raw_tokens": rt,
            "total_compressed_tokens": ct,
            "total_tokens_saved": saved,
            "compression_ratio_pct": round((saved / rt * 100) if rt else 0, 2),
            "avg_proxy_latency_ms": round(row["pl"] or 0, 2),
            "avg_upstream_latency_ms": round(row["ul"] or 0, 2),
            "cache_hits": row["ch"] or 0,
            "ccr_fetches": row["cu"] or 0,
            "algo_breakdown": algo_totals,
            "component_breakdown": component_breakdown,
            "timeline": [
                {
                    "ts": r["ts"],
                    "raw": r["raw_tokens"],
                    "compressed": r["compressed_tokens"],
                }
                for r in reversed(recent_rows)
            ],
        }

    # ── File version tracker ────────────────────────────────────────────────────

    async def file_version_get_latest(self, file_path: str) -> dict[str, Any] | None:
        """Return the most recent stored version of *file_path*, or None."""
        assert self._db
        row = await (
            await self._db.execute(
                "SELECT version, content_hash, content FROM file_versions "
                "WHERE file_path = ? ORDER BY version DESC LIMIT 1",
                (file_path,),
            )
        ).fetchone()
        if row is None:
            return None
        return {
            "version": row["version"],
            "content_hash": row["content_hash"],
            "content": row["content"],
        }

    async def file_version_put(
        self, file_path: str, content_hash: str, content: str, version: int = 1
    ) -> None:
        """Store a new version of a file (or the first version, version=1)."""
        assert self._db
        await self._db.execute(
            "INSERT OR REPLACE INTO file_versions(file_path, version, content_hash, "
            "content, created_at) VALUES (?, ?, ?, ?, ?)",
            (file_path, version, content_hash, content, time.time()),
        )
        await self._db.commit()

    async def file_version_count(self) -> int:
        assert self._db
        row = await (
            await self._db.execute(
                "SELECT COUNT(DISTINCT file_path) as n FROM file_versions"
            )
        ).fetchone()
        return row["n"]

    # ── Cross-component ledger ───────────────────────────────────────────────────

    async def xcomp_lookup(self, run_id: str, content_hash: str) -> dict[str, Any] | None:
        """Check whether *content_hash* was already seen in this run, by whom."""
        assert self._db
        row = await (
            await self._db.execute(
                "SELECT component_id FROM xcomp_ledger WHERE run_id = ? AND content_hash = ?",
                (run_id, content_hash),
            )
        ).fetchone()
        return {"component_id": row["component_id"]} if row else None

    async def xcomp_record(self, run_id: str, content_hash: str, component_id: str) -> None:
        assert self._db
        await self._db.execute(
            "INSERT OR IGNORE INTO xcomp_ledger(run_id, content_hash, component_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (run_id, content_hash, component_id, time.time()),
        )
        await self._db.commit()

    # ── PII log ───────────────────────────────────────────────────────────────

    async def log_pii(self, pattern: str, count: int = 1) -> None:
        assert self._db
        await self._db.execute(
            "INSERT INTO pii_log(ts, pattern, count) VALUES (?, ?, ?)",
            (time.time(), pattern, count),
        )
        await self._db.commit()