"""
JSON Smart Crusher

Converts nested JSON logs, structural configurations, and tool schemas into
specialised flat string representations. Uses a Kneedle-inspired algorithm to
detect where semantic information density plateaus and clips repetitive or
bloated tail entries.

Strategies applied (in order):
  1. Depth clamping – truncate nesting beyond `max_depth`.
  2. Array truncation – keep first N + last 1 element of long arrays.
  3. Repetitive-key deduplication – collapse identical sibling values.
  4. Kneedle density scan – detect and drop tail content past the info knee.
  5. Flat-string reserialisation – compact whitespace.
"""

from __future__ import annotations

import json
import re
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Depth clamping
# ──────────────────────────────────────────────────────────────────────────────

_TRUNCATED = "…"


def _clamp_depth(obj: Any, max_depth: int, _current: int = 0) -> Any:
    if _current >= max_depth:
        if isinstance(obj, dict):
            return {k: _TRUNCATED for k in list(obj.keys())[:3]}
        if isinstance(obj, list):
            return [_TRUNCATED] if obj else []
        return obj

    if isinstance(obj, dict):
        return {k: _clamp_depth(v, max_depth, _current + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clamp_depth(v, max_depth, _current + 1) for v in obj]
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# Array truncation
# ──────────────────────────────────────────────────────────────────────────────

_ARRAY_HEAD = 5
_ARRAY_TAIL = 1


def _truncate_arrays(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _truncate_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        truncated = [_truncate_arrays(v) for v in obj]
        if len(truncated) > _ARRAY_HEAD + _ARRAY_TAIL:
            omitted = len(truncated) - _ARRAY_HEAD - _ARRAY_TAIL
            return truncated[:_ARRAY_HEAD] + [f"… ({omitted} more)"] + truncated[-_ARRAY_TAIL:]
        return truncated
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# Repetitive-key deduplication
# ──────────────────────────────────────────────────────────────────────────────

def _dedup_repeating_values(obj: Any) -> Any:
    """
    Inside a list of dicts, if the same key consistently maps to the same
    value across all elements, hoist it out as a note and strip from elements.
    """
    if isinstance(obj, dict):
        return {k: _dedup_repeating_values(v) for k, v in obj.items()}

    if isinstance(obj, list) and len(obj) > 2:
        # Check if all elements are dicts
        if all(isinstance(el, dict) for el in obj):
            # Find keys with identical values across all elements
            all_keys = set(obj[0].keys()) if obj else set()
            constant_keys = {}
            for key in all_keys:
                vals = [el.get(key) for el in obj]
                if len(set(json.dumps(v, default=str) for v in vals)) == 1:
                    constant_keys[key] = vals[0]

            if constant_keys:
                stripped = [
                    {k: v for k, v in el.items() if k not in constant_keys}
                    for el in obj
                ]
                return {
                    "__constant_fields": constant_keys,
                    "__items": [_dedup_repeating_values(el) for el in stripped],
                }

        return [_dedup_repeating_values(el) for el in obj]

    return obj


# ──────────────────────────────────────────────────────────────────────────────
# Kneedle-inspired information density scan
# ──────────────────────────────────────────────────────────────────────────────

def _unique_key_density(obj: Any) -> list[float]:
    """
    Flatten a JSON object into a list of (level, #unique_keys_at_level).
    Returns a density curve: unique_keys / total_keys per depth level.
    """
    level_keys: dict[int, list[str]] = {}

    def _walk(node: Any, depth: int) -> None:
        if isinstance(node, dict):
            if depth not in level_keys:
                level_keys[depth] = []
            level_keys[depth].extend(node.keys())
            for v in node.values():
                _walk(v, depth + 1)
        elif isinstance(node, list):
            for v in node:
                _walk(v, depth + 1)

    _walk(obj, 0)

    densities: list[float] = []
    for d in sorted(level_keys):
        keys = level_keys[d]
        density = len(set(keys)) / max(len(keys), 1)
        densities.append(density)
    return densities


def _find_knee(densities: list[float]) -> int:
    """
    Simple elbow detection: find the depth level where density drops below
    a running threshold, indicating diminishing returns.
    """
    if len(densities) < 3:
        return len(densities)

    # Normalise
    max_d = max(densities)
    if max_d == 0:
        return len(densities)
    norm = [d / max_d for d in densities]

    # Find first point where density drops > 50% from peak
    threshold = 0.30
    for i, d in enumerate(norm):
        if i > 0 and d < threshold:
            return i
    return len(densities)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def crush_json(
    data: str | dict | list,
    *,
    max_depth: int = 4,
    use_kneedle: bool = True,
) -> tuple[str, int]:
    """
    Crush a JSON value into a compact representation.

    Parameters
    ----------
    data        Raw JSON string, dict, or list.
    max_depth   Hard depth cap before truncation.
    use_kneedle Enable automatic depth detection via Kneedle algorithm.

    Returns
    -------
    (compact_json_string, tokens_saved_estimate)
    """
    if isinstance(data, str):
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return data, 0  # not valid JSON
        original_str = data
    else:
        obj = data
        original_str = json.dumps(data, separators=(",", ":"))

    # Auto-detect effective depth via Kneedle
    if use_kneedle:
        densities = _unique_key_density(obj)
        knee = _find_knee(densities)
        effective_depth = min(max_depth, max(1, knee))
    else:
        effective_depth = max_depth

    obj = _clamp_depth(obj, effective_depth)
    obj = _truncate_arrays(obj)
    obj = _dedup_repeating_values(obj)

    compact = json.dumps(obj, separators=(",", ":"), default=str)
    saved = max(0, len(original_str) - len(compact))
    return compact, int(saved / 3.5)


def crush_json_in_text(text: str, max_depth: int = 4) -> tuple[str, int]:
    """
    Find all embedded JSON objects/arrays in *text* and crush each one.
    Returns (modified_text, total_tokens_saved).
    """
    total_saved = 0

    # Match top-level {...} and [...] blocks (simple non-nested for speed)
    _JSON_BLOCK_RE = re.compile(r"(\{[\s\S]{50,}\}|\[[\s\S]{50,}\])", re.MULTILINE)

    def _replace(m: re.Match) -> str:
        nonlocal total_saved
        crushed, saved = crush_json(m.group(0), max_depth=max_depth)
        total_saved += saved
        return crushed

    return _JSON_BLOCK_RE.sub(_replace, text), total_saved
