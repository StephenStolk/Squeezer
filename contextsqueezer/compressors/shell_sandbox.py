"""
Dynamic Shell Output Sandbox

Intercepts massive stdout/stderr command payloads and applies:
  1. Passing-test line removal (pytest, jest, mocha, go test, cargo test).
  2. Levenshtein-based error grouping – near-identical error lines are
     collapsed into one representative line + a count badge.
  3. Long stack-frame collapsing – chains of stdlib/venv frames are replaced
     with a single `… N stdlib frames …` marker.
  4. Root exception extraction – ensures the final exception/error signature
     always survives, even in very aggressive compression.
"""

from __future__ import annotations

import re
from itertools import groupby

try:
    from Levenshtein import distance as lev_distance  # type: ignore
except ImportError:
    def lev_distance(a: str, b: str) -> int:  # type: ignore[misc]
        """Pure-Python Levenshtein fallback (O(m·n))."""
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n]


# ──────────────────────────────────────────────────────────────────────────────
# Passing-test line patterns
# ──────────────────────────────────────────────────────────────────────────────

_PASS_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*(PASSED|PASS|ok|\.)\s*$", re.IGNORECASE),
    re.compile(r"^\s*test_\w+\s+PASSED", re.IGNORECASE),
    re.compile(r"(?i)\bPASSED\s*$"),                  # pytest -v: "path.py::test PASSED"
    re.compile(r"^\s*✓\s+"),                         # jest / mocha
    re.compile(r"^\s*--- PASS:", re.IGNORECASE),     # go test
    re.compile(r"^\s*test .* \.\.\. ok$", re.IGNORECASE),  # cargo
    re.compile(r"^\s*\d+ passed"),                   # pytest summary line (keep but compact)
    re.compile(r"^\s*collecting \.\.\.", re.IGNORECASE),
]


def _is_passing_line(line: str) -> bool:
    return any(p.search(line) for p in _PASS_PATTERNS)


# ──────────────────────────────────────────────────────────────────────────────
# Stdlib / venv frame patterns
# ──────────────────────────────────────────────────────────────────────────────

_STDLIB_FRAME_RE = re.compile(
    r'File "(?:.*(site-packages|dist-packages|Lib[/\\]|lib/python|<frozen)[^"]*)"'
)
_FRAME_RE = re.compile(r'^\s*File "', re.MULTILINE)


# ──────────────────────────────────────────────────────────────────────────────
# Error / exception root patterns
# ──────────────────────────────────────────────────────────────────────────────

_EXCEPTION_RE = re.compile(
    r"^(?:\w+\.)*\w*(?:Error|Exception|Warning|Failure|Fault|Panic|FAIL|fatal).*",
    re.IGNORECASE | re.MULTILINE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Core processing functions
# ──────────────────────────────────────────────────────────────────────────────

def _remove_passing_tests(lines: list[str]) -> list[str]:
    return [ln for ln in lines if not _is_passing_line(ln)]


def _group_near_identical_errors(
    lines: list[str],
    max_dist_ratio: float = 0.15,
) -> list[str]:
    """
    Group consecutive lines that are near-identical (Levenshtein distance ≤
    max_dist_ratio * len(line)) and collapse them to one representative + count.
    """
    if not lines:
        return lines

    result: list[str] = []
    i = 0
    while i < len(lines):
        current = lines[i]
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            max_len = max(len(current), len(nxt), 1)
            threshold = int(max_len * max_dist_ratio)
            if lev_distance(current.strip(), nxt.strip()) <= threshold:
                j += 1
            else:
                break
        count = j - i
        if count > 1:
            result.append(f"{current.rstrip()} [×{count}]")
        else:
            result.append(current)
        i = j

    return result


def _collapse_stdlib_frames(lines: list[str], keep_top: int = 3) -> list[str]:
    """
    Collapse consecutive stdlib/venv stack frames into a single summary line.
    Always keeps `keep_top` user frames.
    """
    result: list[str] = []
    stdlib_run: list[str] = []

    def flush_stdlib() -> None:
        n = len(stdlib_run)
        if n:
            result.append(f"    … {n} stdlib/venv frame{'s' if n > 1 else ''} omitted …\n")
            stdlib_run.clear()

    for line in lines:
        if _STDLIB_FRAME_RE.search(line):
            stdlib_run.append(line)
        else:
            flush_stdlib()
            result.append(line)
    flush_stdlib()
    return result


def _extract_root_exception(text: str) -> str | None:
    """Return the last (deepest) exception line in the output."""
    matches = _EXCEPTION_RE.findall(text)
    return matches[-1] if matches else None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

_MAX_OUTPUT_LINES = 200  # hard cap after all reductions


def minify_shell_output(
    text: str,
    *,
    max_lines: int = _MAX_OUTPUT_LINES,
) -> tuple[str, int]:
    """
    Compress a shell / terminal output blob.

    Returns (minified_text, tokens_saved_estimate).

    The root exception is always preserved, even under aggressive truncation.
    """
    original_len = len(text)

    # Preserve root exception before we start cutting
    root_exc = _extract_root_exception(text)

    lines = text.splitlines(keepends=True)
    lines = _remove_passing_tests(lines)
    lines = _group_near_identical_errors(lines)
    lines = _collapse_stdlib_frames(lines)

    # Hard line cap
    if len(lines) > max_lines:
        head = lines[: max_lines // 2]
        tail = lines[-(max_lines // 2) :]
        omitted = len(lines) - max_lines
        lines = head + [f"\n… [{omitted} lines omitted] …\n"] + tail

    result = "".join(lines)

    # Guarantee root exception survives
    if root_exc and root_exc not in result:
        result += f"\n[ROOT EXCEPTION] {root_exc}\n"

    saved = max(0, original_len - len(result))
    return result, int(saved / 3.5)
