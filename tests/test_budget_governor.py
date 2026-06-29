from contextsqueezer.config import Settings
from contextsqueezer.pipeline.budget_governor import (
    plan_for_budget,
    intersect_with_settings,
    full_plan,
)


def test_no_budget_returns_full_plan():
    plan = plan_for_budget(raw_tokens=10000, budget_tokens=None)
    assert plan == full_plan()


def test_already_under_budget_uses_light_tier():
    plan = plan_for_budget(raw_tokens=1000, budget_tokens=5000)
    assert plan.tier_reached == 1
    assert plan.enable_ast is False


def test_tight_budget_escalates_to_heaviest_tier():
    plan = plan_for_budget(raw_tokens=100_000, budget_tokens=1000)
    assert plan.tier_reached == 5
    assert plan.enable_temporal is True


def test_moderate_budget_picks_middle_tier():
    # raw=10000, budget=6000 -> needs ~40% reduction, tier 2 ceiling is 35%, tier3 is 55%
    plan = plan_for_budget(raw_tokens=10000, budget_tokens=6000)
    assert plan.tier_reached >= 2


def test_intersect_respects_settings_ceiling():
    settings = Settings(enable_ast_compactor=False)
    plan = plan_for_budget(raw_tokens=100_000, budget_tokens=1000)  # heaviest tier
    intersected = intersect_with_settings(plan, settings)
    # Even though the budget plan wants AST stripping on, settings forbids it.
    assert intersected.enable_ast is False


def test_intersect_ccr_threshold_is_minimum_of_both():
    settings = Settings(ccr_token_threshold=3000)
    plan = plan_for_budget(raw_tokens=100_000, budget_tokens=1000)  # tier 5, ccr_threshold=500
    intersected = intersect_with_settings(plan, settings)
    assert intersected.ccr_threshold == 500
