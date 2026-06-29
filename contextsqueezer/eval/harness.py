"""
Eval Harness — measure real compression impact on real (or recorded) traffic.

This is the concrete answer to "how do I test this against real stuff":

  1. Get real data. Either:
       a. Run the proxy with recording on and point your ACTUAL agent at it
          for a normal session:
              SQUEEZER_ENABLE_RECORDING=true squeezer start
          Every raw, pre-compression request gets appended as one JSON line
          to ~/.config/contextsqueezer/recordings/raw_requests.jsonl.
       b. Or hand-build / export a JSONL file of Anthropic-messages-format
          payloads yourself — see eval/fixtures/sample_coding_session.jsonl
          for the expected shape.

  2. Run the offline report (no API calls, no cost):
         squeezer eval run path/to/file.jsonl

  3. Optionally add --live (requires ANTHROPIC_API_KEY) to replay a sample
     of requests against the REAL API both with and without compression, and
     get a lexical-similarity score between the two answers. This uses
     difflib — no embeddings, no judge model, deliberately simple — good
     enough to flag "compression clearly changed the answer," not a
     rigorous semantic-equivalence proof. Treat a low score as "go read
     both transcripts," not as a final verdict.
"""

from __future__ import annotations

import difflib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from contextsqueezer.config import Settings
from contextsqueezer.pipeline.orchestrator import run_pipeline
from contextsqueezer.storage.sqlite_store import Store, init_db


@dataclass
class CaseResult:
    index: int
    raw_tokens: int
    compressed_tokens: int
    algo_breakdown: dict[str, int]
    proxy_latency_ms: float
    error: str | None = None
    live_similarity: float | None = None
    live_raw_answer: str | None = None
    live_compressed_answer: str | None = None

    @property
    def tokens_saved(self) -> int:
        return self.raw_tokens - self.compressed_tokens

    @property
    def compression_pct(self) -> float:
        return (self.tokens_saved / self.raw_tokens * 100) if self.raw_tokens else 0.0


@dataclass
class EvalReport:
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def total_raw_tokens(self) -> int:
        return sum(c.raw_tokens for c in self.cases)

    @property
    def total_compressed_tokens(self) -> int:
        return sum(c.compressed_tokens for c in self.cases)

    @property
    def total_saved(self) -> int:
        return self.total_raw_tokens - self.total_compressed_tokens

    @property
    def overall_compression_pct(self) -> float:
        return (self.total_saved / self.total_raw_tokens * 100) if self.total_raw_tokens else 0.0

    @property
    def algo_totals(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for c in self.cases:
            for k, v in c.algo_breakdown.items():
                totals[k] = totals.get(k, 0) + v
        return totals

    @property
    def avg_proxy_latency_ms(self) -> float:
        vals = [c.proxy_latency_ms for c in self.cases if c.error is None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def low_similarity_cases(self) -> list[CaseResult]:
        return [c for c in self.cases if c.live_similarity is not None and c.live_similarity < 0.5]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": len(self.cases),
            "total_raw_tokens": self.total_raw_tokens,
            "total_compressed_tokens": self.total_compressed_tokens,
            "total_tokens_saved": self.total_saved,
            "overall_compression_pct": round(self.overall_compression_pct, 2),
            "avg_proxy_latency_ms": round(self.avg_proxy_latency_ms, 2),
            "algo_totals": self.algo_totals,
            "low_similarity_case_indices": [c.index for c in self.low_similarity_cases],
            "cases": [
                {
                    "index": c.index,
                    "raw_tokens": c.raw_tokens,
                    "compressed_tokens": c.compressed_tokens,
                    "compression_pct": round(c.compression_pct, 2),
                    "algo_breakdown": c.algo_breakdown,
                    "error": c.error,
                    "live_similarity": c.live_similarity,
                }
                for c in self.cases
            ],
        }


def _unwrap_payload(entry: dict) -> dict:
    """Recorded entries are wrapped as {"path":..., "ts":..., "payload": {...}}."""
    if "payload" in entry and "messages" not in entry:
        return entry["payload"]
    return entry


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    payloads: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            payloads.append(_unwrap_payload(entry))
            if limit and len(payloads) >= limit:
                break
    return payloads


async def _call_anthropic_live(
    payload: dict, api_key: str, timeout: float = 60.0
) -> str:
    """Make a real call to the Anthropic Messages API and return the text reply."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        texts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        return "\n".join(texts)


def _similarity(a: str, b: str) -> float:
    """Deterministic lexical similarity — no embeddings, no judge model."""
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


async def run_eval(
    payloads: list[dict],
    *,
    settings: Settings,
    store: Store,
    live: bool = False,
    api_key: str | None = None,
) -> EvalReport:
    """Run the compression pipeline (and optionally live API calls) over *payloads*."""
    report = EvalReport()

    for i, payload in enumerate(payloads):
        try:
            result = await run_pipeline(payload, settings=settings, store=store)
        except Exception as exc:  # keep going — one bad case shouldn't kill the run
            report.cases.append(
                CaseResult(index=i, raw_tokens=0, compressed_tokens=0, algo_breakdown={},
                           proxy_latency_ms=0.0, error=str(exc))
            )
            continue

        case = CaseResult(
            index=i,
            raw_tokens=result.raw_tokens,
            compressed_tokens=result.compressed_tokens,
            algo_breakdown=result.algo_breakdown,
            proxy_latency_ms=result.proxy_latency_ms,
        )

        if live and api_key:
            try:
                raw_answer = await _call_anthropic_live(payload, api_key)
                compressed_answer = await _call_anthropic_live(
                    result.compressed_payload, api_key
                )
                case.live_raw_answer = raw_answer
                case.live_compressed_answer = compressed_answer
                case.live_similarity = _similarity(raw_answer, compressed_answer)
            except Exception as exc:
                case.error = f"live call failed: {exc}"

        report.cases.append(case)

    return report


async def run_eval_from_file(
    path: Path,
    *,
    settings: Settings,
    live: bool = False,
    limit: int | None = None,
) -> EvalReport:
    payloads = load_jsonl(path, limit=limit)
    await init_db(settings.db_path)
    api_key = os.environ.get("ANTHROPIC_API_KEY") if live else None

    async with Store(settings.db_path) as store:
        return await run_eval(payloads, settings=settings, store=store, live=live, api_key=api_key)
