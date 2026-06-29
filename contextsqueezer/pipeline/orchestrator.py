"""
Pipeline Orchestrator

End-to-end sequential pipeline for processing intercepted LLM request payloads.

Step 1  – Squeezer-metadata extraction (component_id / run_id / budget_tokens)
Step 2  – Budget governor (pick compression aggressiveness tier)
Step 3  – Zero-trust PII scrubber
Step 4  – LSH cross-turn deduplication
Step 5  – Temporal context decay (pin-aware)
Step 6  – Per-message reduction engines (AST, call-graph, file-version, JSON,
           shell, linguistic) + cross-component dedup + CCR offload
Step 7  – Cache alignment (stable-prefix safe)
Step 8  – CCR tool injection
Step 9  – Egress (handled by proxy server)

All steps track per-algorithm token savings for dashboard telemetry.
"""

from __future__ import annotations

import asyncio
import copy
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from contextsqueezer.config import Settings
from contextsqueezer.compressors.ast_compactor import compact_code
from contextsqueezer.compressors.file_version_tracker import FileVersionTracker
from contextsqueezer.compressors.json_crusher import crush_json, crush_json_in_text
from contextsqueezer.compressors.linguistic_minifier import minify_text
from contextsqueezer.compressors.lsh_deduplicator import deduplicate_turns
from contextsqueezer.compressors.shell_sandbox import minify_shell_output
from contextsqueezer.compressors.temporal_decay import apply_temporal_decay
from contextsqueezer.pipeline import budget_governor
from contextsqueezer.pipeline.cache_aligner import align_for_cache
from contextsqueezer.pipeline.classifier import ContentKind, classify_messages, split_into_blocks
from contextsqueezer.pipeline.component_router import ComponentRouter, SqueezerMeta, extract_meta
from contextsqueezer.security.pii_scrubber import PiiScrubber
from contextsqueezer.storage.ccr import CCRManager
from contextsqueezer.storage.sqlite_store import Store

_SCRUBBER = PiiScrubber()

# File-path hint pattern – agents often include `// path/to/file.py` headers
_FILE_PATH_HINT_RE = re.compile(r"(?:^|//|#)\s*([\w\-./]+\.[a-z]{2,6})\s*$", re.MULTILINE)


@dataclass
class PipelineResult:
    request_id: str
    original_payload: dict
    compressed_payload: dict
    raw_tokens: int
    compressed_tokens: int
    algo_breakdown: dict[str, int] = field(default_factory=dict)
    pii_hits: dict[str, int] = field(default_factory=dict)
    ccr_used: bool = False
    proxy_latency_ms: float = 0.0
    component_id: str = ""
    run_id: str = ""
    budget_tier: int = -1

    @property
    def tokens_saved(self) -> int:
        return self.raw_tokens - self.compressed_tokens

    @property
    def compression_pct(self) -> float:
        if self.raw_tokens == 0:
            return 0.0
        return self.tokens_saved / self.raw_tokens * 100


def _estimate_tokens(payload: dict) -> int:
    """Rough token count of entire payload (≈4 chars/token heuristic)."""
    import json

    text = json.dumps(payload, default=str)
    return max(1, len(text) // 4)


def _extract_file_hint(text: str) -> str | None:
    m = _FILE_PATH_HINT_RE.search(text)
    return m.group(1) if m else None


def _get_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text", b.get("content", "")))
        return "\n".join(str(p) for p in parts)
    return str(content)


def _set_text(msg: dict, new_text: str) -> dict:
    content = msg.get("content", "")
    if isinstance(content, str):
        return {**msg, "content": new_text}
    if isinstance(content, list):
        new_blocks = []
        replaced = False
        for b in content:
            if not replaced and isinstance(b, dict) and b.get("type") == "text":
                new_blocks.append({**b, "text": new_text})
                replaced = True
            else:
                new_blocks.append(b)
        return {**msg, "content": new_blocks}
    return {**msg, "content": new_text}


# ──────────────────────────────────────────────────────────────────────────────
# Per-message compressor
# ──────────────────────────────────────────────────────────────────────────────

async def _compress_block(
    text: str,
    kind: ContentKind,
    settings: Settings,
    plan: budget_governor.CompressionPlan | None,
    file_tracker: FileVersionTracker,
) -> tuple[str, dict[str, int]]:
    """
    Apply content-type-specific compression to a single block of text (which
    may be the *entire* message content, or one sub-block of a MIXED message
    after splitting). This is the part of compression that's purely a
    function of (text, kind) — cross-component dedup and CCR offload live one
    level up in `_compress_message` because they're message-scoped concerns.
    """
    breakdown: dict[str, int] = {}

    enable_ast = plan.enable_ast if plan else settings.enable_ast_compactor
    enable_json = plan.enable_json if plan else settings.enable_json_crusher
    enable_shell = plan.enable_shell if plan else settings.enable_shell_sandbox
    enable_linguistic = plan.enable_linguistic if plan else settings.enable_linguistic_minifier

    # ── Code content: file-version delta, then AST compaction ─────────────────
    if kind == ContentKind.CODE:
        file_hint = _extract_file_hint(text)
        skip_ast = False

        if file_hint and settings.enable_file_version_tracker:
            version_result = await file_tracker.process(file_hint, text)
            if version_result.is_delta:
                breakdown["file_version_tracker"] = (
                    breakdown.get("file_version_tracker", 0) + version_result.tokens_saved
                )
                text = version_result.text
                skip_ast = True  # already a diff/pointer — nothing left to compact

        if enable_ast and not skip_ast:
            compacted, saved = compact_code(text, file_path=file_hint or "")
            if saved > 0:
                breakdown["ast_compactor"] = breakdown.get("ast_compactor", 0) + saved
                text = compacted

    # ── JSON content ──────────────────────────────────────────────────────────
    if kind == ContentKind.JSON_DATA and enable_json:
        crushed, saved = crush_json(text, max_depth=settings.json_max_depth)
        if saved > 0:
            breakdown["json_crusher"] = breakdown.get("json_crusher", 0) + saved
            text = crushed
    elif enable_json:
        modified, saved = crush_json_in_text(text, max_depth=settings.json_max_depth)
        if saved > 0:
            breakdown["json_crusher"] = breakdown.get("json_crusher", 0) + saved
            text = modified

    # ── Shell output ──────────────────────────────────────────────────────────
    if kind == ContentKind.SHELL_OUTPUT and enable_shell:
        minified, saved = minify_shell_output(text)
        if saved > 0:
            breakdown["shell_sandbox"] = breakdown.get("shell_sandbox", 0) + saved
            text = minified

    # ── Linguistic minification ───────────────────────────────────────────────
    if enable_linguistic and kind == ContentKind.CONVERSATION:
        minified_text, saved = minify_text(text, strip_md=False)
        if saved > 0:
            breakdown["linguistic_minifier"] = breakdown.get("linguistic_minifier", 0) + saved
            text = minified_text

    return text, breakdown


async def _compress_message(
    msg: dict,
    kind: ContentKind,
    settings: Settings,
    plan: budget_governor.CompressionPlan | None,
    ccr: CCRManager,
    file_tracker: FileVersionTracker,
    component_router: ComponentRouter,
) -> tuple[dict, dict[str, int]]:
    """Apply content-type-specific compression to a single message."""
    breakdown: dict[str, int] = {}
    content = _get_text(msg.get("content", ""))

    if not content.strip():
        return msg, breakdown

    # ── Cross-component dedup (cheapest possible win — check first) ───────────
    if settings.enable_component_router and component_router.enabled:
        deduped, xsaved = await component_router.dedupe_against_run(content)
        if xsaved > 0:
            breakdown["xcomp_dedup"] = breakdown.get("xcomp_dedup", 0) + xsaved
            return _set_text(msg, deduped), breakdown
        content = deduped  # unchanged if not a dupe, but keeps the flow linear

    if kind == ContentKind.MIXED:
        # A single message spanning multiple content types (e.g. prose with
        # an embedded shell trace) — split and compress each sub-block by its
        # OWN kind instead of letting the whole message fall through every
        # type-specific branch untouched.
        blocks = split_into_blocks(content)
        compressed_parts: list[str] = []
        for block in blocks:
            part_text, part_bd = await _compress_block(
                block.text, block.kind, settings, plan, file_tracker
            )
            compressed_parts.append(part_text)
            for k, v in part_bd.items():
                breakdown[k] = breakdown.get(k, 0) + v
        content = "".join(compressed_parts)
    else:
        content, block_bd = await _compress_block(content, kind, settings, plan, file_tracker)
        for k, v in block_bd.items():
            breakdown[k] = breakdown.get(k, 0) + v

    # ── CCR offload (message-scoped — runs after all block-level work) ───────
    if settings.enable_ccr:
        pointer = await ccr.maybe_offload(content, label=kind.name.lower())
        if pointer != content:
            saved_ccr = len(content) // 4 - len(pointer) // 4
            breakdown["ccr"] = breakdown.get("ccr", 0) + max(0, saved_ccr)
            content = pointer

    return _set_text(msg, content), breakdown


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

async def run_pipeline(
    payload: dict,
    *,
    settings: Settings,
    store: Store,
    headers: dict | None = None,
) -> PipelineResult:
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]
    algo_breakdown: dict[str, int] = {}

    working = copy.deepcopy(payload)

    # ── Step 1: Squeezer metadata extraction (pops squeezer_meta from body) ───
    meta: SqueezerMeta = extract_meta(working, headers or {})

    messages: list[dict] = working.get("messages", [])
    tools: list[dict] = working.get("tools", [])
    raw_tokens = _estimate_tokens(working)

    # ── Step 2: Budget governor ────────────────────────────────────────────────
    plan: budget_governor.CompressionPlan | None = None
    if meta.budget_tokens:
        raw_plan = budget_governor.plan_for_budget(raw_tokens, meta.budget_tokens)
        plan = budget_governor.intersect_with_settings(raw_plan, settings)

    # ── Step 3: PII scrubber (always on if enabled — never gated by budget) ───
    if settings.enable_pii_scrubber:
        messages, pii_hits = _SCRUBBER.scrub_messages(messages)
        if pii_hits:
            asyncio.create_task(_log_pii_hits(store, pii_hits))
    else:
        pii_hits = {}

    # ── Step 4: LSH cross-turn deduplication ──────────────────────────────────
    enable_lsh = plan.enable_lsh if plan else settings.enable_lsh_deduplicator
    if enable_lsh:
        messages, saved = deduplicate_turns(
            messages, similarity_threshold=settings.lsh_similarity_threshold
        )
        if saved:
            algo_breakdown["lsh_deduplicator"] = saved

    # ── Step 5: Temporal decay (pin-aware) ────────────────────────────────────
    enable_temporal = plan.enable_temporal if plan else settings.enable_temporal_decay
    if enable_temporal:
        messages, saved = apply_temporal_decay(
            messages,
            recent_turns=settings.temporal_recent_turns,
            partial_turns=settings.temporal_partial_turns,
        )
        if saved:
            algo_breakdown["temporal_decay"] = saved

    # ── Step 6: Per-message content compression (parallel) ───────────────────
    effective_ccr_threshold = plan.ccr_threshold if plan else settings.ccr_token_threshold
    ccr = CCRManager(store, token_threshold=effective_ccr_threshold)
    file_tracker = FileVersionTracker(store, diff_threshold_ratio=settings.file_version_diff_threshold)
    component_router = ComponentRouter(store, meta, ccr=ccr)
    classified = classify_messages(messages)

    compress_tasks = [
        _compress_message(msg, kind, settings, plan, ccr, file_tracker, component_router)
        for msg, kind in classified
    ]
    results = await asyncio.gather(*compress_tasks)

    compressed_messages: list[dict] = []
    for new_msg, bd in results:
        compressed_messages.append(new_msg)
        for k, v in bd.items():
            algo_breakdown[k] = algo_breakdown.get(k, 0) + v

    messages = compressed_messages

    # ── Step 7: Cache alignment ───────────────────────────────────────────────
    if settings.enable_cache_aligner:
        messages, tools, _ = align_for_cache(
            messages, tools, provider=_detect_provider(payload)
        )

    # ── Step 8: CCR tool injection ────────────────────────────────────────────
    if settings.enable_ccr and ccr.was_used:
        tools = ccr.inject_tool(tools or [])

    # Assemble final payload
    working["messages"] = messages
    if tools:
        working["tools"] = tools
    elif "tools" in working and not tools:
        del working["tools"]

    compressed_tokens = _estimate_tokens(working)
    proxy_latency_ms = (time.perf_counter() - t0) * 1000

    return PipelineResult(
        request_id=request_id,
        original_payload=payload,
        compressed_payload=working,
        raw_tokens=raw_tokens,
        compressed_tokens=compressed_tokens,
        algo_breakdown=algo_breakdown,
        pii_hits=pii_hits,
        ccr_used=ccr.was_used,
        proxy_latency_ms=proxy_latency_ms,
        component_id=meta.component_id or "",
        run_id=meta.run_id or "",
        budget_tier=plan.tier_reached if plan else -1,
    )


async def _log_pii_hits(store: Store, hits: dict[str, int]) -> None:
    for pattern, count in hits.items():
        try:
            await store.log_pii(pattern, count)
        except Exception:
            pass


def _detect_provider(payload: dict) -> str:
    model = payload.get("model", "")
    if "claude" in model.lower():
        return "anthropic"
    if "gpt" in model.lower() or "o1" in model.lower():
        return "openai"
    return "generic"
