"""
Budget Governor — declarative, adaptive compression intensity.

Headroom and most rule-based compressors apply one fixed level of
aggressiveness to every request, all the time. That's needlessly destructive
for a request that's already small, and it's not adaptive to a caller who
knows their actual constraint ("this call has to fit in an 8K-token slot
because it's going into a sub-agent with a small window").

The budget governor lets any caller declare a target token budget for a
*specific* call — via `squeezer_meta.budget_tokens` or the
`X-Squeezer-Budget` header — and the pipeline escalates through
aggressiveness tiers only as far as actually needed:

  Tier 0 — PII scrub + exact dedup only (always on; never skipped)
  Tier 1 — + LSH near-duplicate dedup across turns
  Tier 2 — + linguistic minification + JSON crushing
  Tier 3 — + AST body stripping + shell-output minification
  Tier 4 — + temporal decay (old turns → keyword digest)
  Tier 5 — + CCR offload threshold dropped aggressively

If no budget is given, the pipeline runs at whatever aggressiveness the
global Settings flags specify (today's default behaviour, unchanged). A
budget can only ever *restrict* — it's ANDed against the global Settings —
so a globally-disabled stage never gets switched on just because a budget
asked for it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompressionPlan:
    enable_lsh: bool
    enable_linguistic: bool
    enable_json: bool
    enable_ast: bool
    enable_shell: bool
    enable_temporal: bool
    ccr_threshold: int
    tier_reached: int


# Tiers ordered lightest → heaviest.
_TIERS: list[CompressionPlan] = [
    CompressionPlan(False, False, False, False, False, False, 999_999, 0),
    CompressionPlan(True, False, False, False, False, False, 999_999, 1),
    CompressionPlan(True, True, True, False, False, False, 4000, 2),
    CompressionPlan(True, True, True, True, True, False, 2000, 3),
    CompressionPlan(True, True, True, True, True, True, 2000, 4),
    CompressionPlan(True, True, True, True, True, True, 500, 5),
]

# Empirically-conservative ceiling on *cumulative* token-reduction achievable
# by each tier, used only to pick a reasonable starting tier — the orchestrator
# still measures actual results afterward; this isn't a correctness guarantee.
_REDUCTION_CEILINGS = [0.05, 0.15, 0.35, 0.55, 0.70, 0.85]


def full_plan() -> CompressionPlan:
    """The most aggressive tier — used when no budget is specified."""
    return _TIERS[-1]


def plan_for_budget(raw_tokens: int, budget_tokens: int | None) -> CompressionPlan:
    """
    Pick the lightest tier whose projected output plausibly fits the budget.
    Falls back to the heaviest tier if even that isn't projected to be enough.
    """
    if budget_tokens is None or budget_tokens <= 0 or raw_tokens <= budget_tokens:
        # Already within budget (or no budget set) — use the lightest tier
        # that still does the "always on" baseline work.
        if budget_tokens is not None and raw_tokens <= budget_tokens:
            return _TIERS[1]
        return full_plan()

    for tier, ceiling in zip(_TIERS, _REDUCTION_CEILINGS):
        projected = raw_tokens * (1 - ceiling)
        if projected <= budget_tokens:
            return tier

    return _TIERS[-1]


def intersect_with_settings(plan: CompressionPlan, settings) -> CompressionPlan:  # type: ignore[no-untyped-def]
    """AND a budget plan against global Settings — settings act as a ceiling."""
    return CompressionPlan(
        enable_lsh=plan.enable_lsh and settings.enable_lsh_deduplicator,
        enable_linguistic=plan.enable_linguistic and settings.enable_linguistic_minifier,
        enable_json=plan.enable_json and settings.enable_json_crusher,
        enable_ast=plan.enable_ast and settings.enable_ast_compactor,
        enable_shell=plan.enable_shell and settings.enable_shell_sandbox,
        enable_temporal=plan.enable_temporal and settings.enable_temporal_decay,
        ccr_threshold=min(plan.ccr_threshold, settings.ccr_token_threshold),
        tier_reached=plan.tier_reached,
    )
