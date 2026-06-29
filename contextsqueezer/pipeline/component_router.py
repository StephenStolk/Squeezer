"""
Component-Aware Context Router

This is the answer to "what if individual components of a bigger system each
fed the proxy before hitting the LLM."

In multi-agent / multi-component systems (a LangGraph graph, a swarm of
cooperating agents, several microservices that each call the same model),
independent components routinely send *overlapping* context — the same
retrieved document chunk surfaces in two different agents' prompts, the same
tool result gets re-explained to a second agent a few seconds later, etc.

Plain in-conversation deduplication (the LSH module) only ever sees one
component's own message history. It has no way to know that a *different*
component already paid to send identical content moments earlier — because
from its point of view, those are two entirely separate, unrelated requests.

Component-aware routing closes that gap. Callers tag each request with:
  • component_id  — which logical agent/service is making this call
  • run_id        — a shared identifier for the overall multi-component
                    session (e.g. a LangGraph run ID, a trace ID)

The proxy keeps a local, run-scoped content-hash ledger. When component B's
payload contains a block whose content hash was already recorded by another
component within the same run, B's copy is stored via the same CCR
mechanism used for size-based offloading and replaced with a real
`[CCR:...]` pointer — retrievable via `squeezer_retrieve` like any other
CCR entry, not a dead-end marker.

Tagging convention (either works, headers take precedence):
  Headers:  X-Squeezer-Component: retriever_agent
            X-Squeezer-Run: langgraph-run-8f3c
            X-Squeezer-Budget: 8000
  Body:     {"squeezer_meta": {"component_id": "...", "run_id": "...",
                                "budget_tokens": 8000}}

The `squeezer_meta` body field, if present, is popped out of the payload
before it's forwarded upstream — providers would reject an unknown field.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextsqueezer.storage.ccr import CCRManager
    from contextsqueezer.storage.sqlite_store import Store

# Don't bother cross-component-deduping trivially short strings — the
# pointer overhead isn't worth it and short strings collide too often
# to be a meaningful redundancy signal.
_MIN_LEN_FOR_XCOMP = 200


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass
class SqueezerMeta:
    component_id: str | None = None
    run_id: str | None = None
    budget_tokens: int | None = None

    @property
    def cross_component_enabled(self) -> bool:
        return bool(self.component_id and self.run_id)


def extract_meta(payload: dict, headers: dict) -> SqueezerMeta:
    """
    Pull squeezer metadata from request headers (preferred) or a
    `squeezer_meta` body field (fallback). Mutates *payload* in place,
    removing `squeezer_meta` so it never reaches the upstream provider.
    """
    body_meta = {}
    if isinstance(payload, dict):
        body_meta = payload.pop("squeezer_meta", {}) or {}

    headers_lower = {k.lower(): v for k, v in (headers or {}).items()}

    def _get(body_key: str, header_name: str) -> str | None:
        return headers_lower.get(header_name) or body_meta.get(body_key)

    budget_raw = _get("budget_tokens", "x-squeezer-budget")
    budget: int | None
    try:
        budget = int(budget_raw) if budget_raw is not None else None
    except (TypeError, ValueError):
        budget = None

    return SqueezerMeta(
        component_id=_get("component_id", "x-squeezer-component"),
        run_id=_get("run_id", "x-squeezer-run"),
        budget_tokens=budget,
    )


class ComponentRouter:
    """Per-request handle for cross-component dedup against the shared ledger."""

    def __init__(self, store: "Store", meta: SqueezerMeta, ccr: "CCRManager | None" = None) -> None:
        self._store = store
        self._meta = meta
        self._ccr = ccr

    @property
    def enabled(self) -> bool:
        return self._meta.cross_component_enabled

    @property
    def component_id(self) -> str:
        return self._meta.component_id or ""

    @property
    def run_id(self) -> str:
        return self._meta.run_id or ""

    async def dedupe_against_run(self, text: str) -> tuple[str, int]:
        """
        If cross-component dedup is enabled and *text* was already recorded
        by a *different* component in this run, return a pointer instead.

        The duplicate's content is stored via the same CCR mechanism used
        for size-based offloading, so the resulting pointer is a real
        `[CCR:...]` pointer the model can resolve with `squeezer_retrieve` —
        not a dead-end marker. Requires a CCRManager to be supplied; without
        one, dedup is skipped entirely (fail safe, not fail silent-and-lossy).

        Returns (possibly-replaced text, tokens_saved).
        """
        if not self.enabled or len(text) < _MIN_LEN_FOR_XCOMP:
            return text, 0

        h = _hash(text)
        seen = await self._store.xcomp_lookup(self._meta.run_id, h)  # type: ignore[arg-type]

        if seen and seen["component_id"] != self._meta.component_id:
            if self._ccr is None:
                # No retrievable backing store available — skip rather than
                # emit a pointer the model could never resolve.
                return text, 0
            pointer = await self._ccr.force_offload(
                text, label=f"xcomp:{seen['component_id']}"
            )
            saved = max(0, len(text) - len(pointer))
            return pointer, int(saved / 3.5)

        if not seen:
            await self._store.xcomp_record(
                self._meta.run_id, h, self._meta.component_id  # type: ignore[arg-type]
            )
        return text, 0
