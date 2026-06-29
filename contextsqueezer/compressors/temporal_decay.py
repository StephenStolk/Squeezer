"""
Temporal Context Decay

Dynamically steps through conversational logs based on age (turn index):

  Turn 0..recent-1         → verbatim, untouched
  Turn recent..partial-1   → strip markdown decoration + padding
  Turn partial+            → condense to raw state keywords only

This simulates a memory-forgetting curve: the agent retains full detail for
recent interactions but only semantic key-phrases for older ones.
"""

from __future__ import annotations

import re
from typing import Sequence

from contextsqueezer.compressors.linguistic_minifier import strip_markdown_decoration

# ──────────────────────────────────────────────────────────────────────────────
# Keyword extraction (fallback – no external NLP dependency)
# ──────────────────────────────────────────────────────────────────────────────

# English stop words (lightweight set)
_STOP_WORDS = frozenset(
    """a about after all also an and any are as at be been before being below
    between both but by can could did do does doing down during each few for
    from further get got had has have he her here him his how i if in into is
    it its itself just keep let like likely may me might more most much must my
    need no nor not now of off on once only or other our out over own per same
    see set should since so some still such than that the their them then there
    these they this those through to too under until up use very via was we
    were what when where which while who will with would you your""".split()
)

_WORD_RE = re.compile(r"[a-zA-Z_][\w]{2,}")   # words ≥ 3 chars
_CODE_SYMBOL_RE = re.compile(r"`([^`]+)`")       # backtick-quoted code symbols
_FILE_PATH_RE = re.compile(r"[\w\-./]+\.[a-z]{2,5}")  # file paths


def _extract_keywords(text: str, max_keywords: int = 25) -> str:
    """
    Extract the most informative terms from *text* and return them as a
    comma-separated keyword string.
    """
    keywords: list[str] = []

    # Always preserve backtick-quoted code symbols (high information density)
    for m in _CODE_SYMBOL_RE.finditer(text):
        sym = m.group(1).strip()
        if sym and sym not in keywords:
            keywords.append(sym)

    # Preserve file paths / module references
    for m in _FILE_PATH_RE.finditer(text):
        p = m.group(0)
        if p not in keywords:
            keywords.append(p)

    # Regular words filtered for stop words
    for m in _WORD_RE.finditer(text):
        w = m.group(0).lower()
        if w not in _STOP_WORDS and w not in keywords:
            keywords.append(m.group(0))

    return ", ".join(keywords[:max_keywords])


# ──────────────────────────────────────────────────────────────────────────────
# Role-aware condensation
# ──────────────────────────────────────────────────────────────────────────────

_ROLE_ABBREV = {"user": "U", "assistant": "A", "system": "SYS", "tool": "TOOL"}


def _condense_to_keywords(msg: dict) -> dict:
    """Reduce a message to a keyword digest."""
    role = msg.get("role", "?")
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", block.get("content", "")))
        content = " ".join(str(p) for p in parts)

    role_abbr = _ROLE_ABBREV.get(role, role[:1].upper())
    keywords = _extract_keywords(str(content))
    return {**msg, "content": f"[{role_abbr}·{keywords}]"}


def _strip_markdown(msg: dict) -> dict:
    """Strip markdown decoration from a message."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return {**msg, "content": strip_markdown_decoration(content)}
    if isinstance(content, list):
        new_blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                new_blocks.append(
                    {**block, "text": strip_markdown_decoration(block.get("text", ""))}
                )
            else:
                new_blocks.append(block)
        return {**msg, "content": new_blocks}
    return msg


# ──────────────────────────────────────────────────────────────────────────────
# Pinning — protect specific turns from age-based decay
# ──────────────────────────────────────────────────────────────────────────────

PIN_MARKER = "[PIN]"


def _message_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    return str(content)


def _is_pinned(msg: dict) -> bool:
    """A message is pinned if its content (or first text block) starts with [PIN]."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.lstrip().startswith(PIN_MARKER)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "").lstrip().startswith(PIN_MARKER)
    return False


def _strip_pin_marker(msg: dict) -> dict:
    """Remove the literal [PIN] marker before the message goes upstream."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return {**msg, "content": content.lstrip().removeprefix(PIN_MARKER).lstrip()}
    if isinstance(content, list):
        new_blocks = []
        replaced = False
        for block in content:
            if not replaced and isinstance(block, dict) and block.get("type") == "text":
                stripped = block.get("text", "").lstrip().removeprefix(PIN_MARKER).lstrip()
                new_blocks.append({**block, "text": stripped})
                replaced = True
            else:
                new_blocks.append(block)
        return {**msg, "content": new_blocks}
    return msg


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def apply_temporal_decay(
    messages: list[dict],
    *,
    recent_turns: int = 2,
    partial_turns: int = 8,
) -> tuple[list[dict], int]:
    """
    Apply the half-life context aging matrix to a messages list.

    Parameters
    ----------
    messages        Full conversation history (oldest first).
    recent_turns    Last N turns kept verbatim.
    partial_turns   Next M turns (before recent) kept with markdown stripped.
                    Everything older is keyword-condensed.

    Returns
    -------
    (processed_messages, tokens_saved_estimate)
    """
    n = len(messages)
    tokens_saved = 0
    out: list[dict] = []

    # Always preserve system prompt (index 0 if role == "system")
    system_indices: set[int] = set()
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            system_indices.add(i)

    for i, msg in enumerate(messages):
        if i in system_indices:
            out.append(msg)
            continue

        if _is_pinned(msg):
            # Pinned turns are exempt from decay regardless of age, but the
            # literal marker is stripped so the model doesn't see it.
            out.append(_strip_pin_marker(msg))
            continue

        # Position from the end (0 = most recent)
        age = n - 1 - i

        original_text = msg.get("content", "")
        if isinstance(original_text, list):
            original_text = " ".join(
                b.get("text", "") for b in original_text if isinstance(b, dict)
            )

        if age < recent_turns:
            # Zone A: verbatim
            out.append(msg)
        elif age < recent_turns + partial_turns:
            # Zone B: strip markdown + padding
            new_msg = _strip_markdown(msg)
            saved = max(0, len(str(original_text)) - len(str(new_msg.get("content", ""))))
            tokens_saved += int(saved / 3.5)
            out.append(new_msg)
        else:
            # Zone C: keyword digest only
            new_msg = _condense_to_keywords(msg)
            saved = max(0, len(str(original_text)) - len(str(new_msg.get("content", ""))))
            tokens_saved += int(saved / 3.5)
            out.append(new_msg)

    return out, tokens_saved
