"""
LSH Cross-Turn Deduplicator

Slices sliding-window buffers across multi-turn text arrays, applies Rabin
rolling-hash fingerprinting to catch duplicate or highly-overlapping
documentation / code blocks, and replaces all but the primary instance with a
short context-pointer token.

Algorithm
---------
  1. Split each message content into fixed-size shingles (character n-grams).
  2. Compute a 64-bit SimHash per shingle set.
  3. Build a sliding window over the turn sequence.
  4. If two turns share >85% (configurable) bit-similarity, mark the later one
     as a duplicate and replace it with a [DEDUP:<turn_idx>] pointer.
  5. Exact Rabin fingerprint check runs first for O(1) exact matches.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Sequence


# ──────────────────────────────────────────────────────────────────────────────
# Rabin fingerprint (exact match)
# ──────────────────────────────────────────────────────────────────────────────

_BASE = 257
_MOD = (1 << 61) - 1  # Mersenne prime


def rabin_fingerprint(text: str) -> int:
    """Fast polynomial rolling hash over the full text."""
    h = 0
    for ch in text:
        h = (h * _BASE + ord(ch)) % _MOD
    return h


# ──────────────────────────────────────────────────────────────────────────────
# SimHash (approximate near-duplicate detection)
# ──────────────────────────────────────────────────────────────────────────────

_BITS = 64
_SHINGLE_SIZE = 4   # character 4-grams
_MIN_TEXT_LEN = 80  # don't bother deduping short snippets


def _shingles(text: str, k: int = _SHINGLE_SIZE) -> list[str]:
    return [text[i : i + k] for i in range(max(0, len(text) - k + 1))]


def _md5_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def simhash(text: str) -> int:
    """Compute a 64-bit SimHash for the given text."""
    shingle_list = _shingles(text)
    if not shingle_list:
        return 0
    counts = [0] * _BITS
    for shingle in shingle_list:
        h = _md5_int(shingle)
        for i in range(_BITS):
            if h & (1 << i):
                counts[i] += 1
            else:
                counts[i] -= 1
    result = 0
    for i in range(_BITS):
        if counts[i] > 0:
            result |= 1 << i
    return result


def hamming_similarity(a: int, b: int) -> float:
    """Normalised bit-similarity between two SimHash values (0.0–1.0)."""
    xor = a ^ b
    differing = bin(xor).count("1")
    return 1.0 - differing / _BITS


# ──────────────────────────────────────────────────────────────────────────────
# Deduplicator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _TurnFingerprint:
    index: int
    text: str
    rabin: int
    simhash_val: int


def _extract_text(message: dict) -> str:
    """Get plain text from a message dict (handles str and block-list content)."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


def deduplicate_turns(
    messages: list[dict],
    *,
    similarity_threshold: float = 0.85,
    window_size: int = 10,
) -> tuple[list[dict], int]:
    """
    Deduplicate near-identical content across conversation turns.

    Returns (processed_messages, tokens_saved_estimate).

    Duplicate turns have their content replaced with a pointer:
        [DEDUP:ref=<original_turn_index> sim=0.97]
    """
    fps: list[_TurnFingerprint] = []
    tokens_saved = 0
    out_messages: list[dict] = []

    for i, msg in enumerate(messages):
        text = _extract_text(msg)

        if len(text) < _MIN_TEXT_LEN:
            out_messages.append(msg)
            fps.append(_TurnFingerprint(i, text, rabin_fingerprint(text), simhash(text)))
            continue

        rb = rabin_fingerprint(text)
        sh = simhash(text)

        # Window: look at the last `window_size` turns
        window_start = max(0, len(fps) - window_size)
        window = fps[window_start:]

        duplicate_of: int | None = None
        best_sim: float = 0.0

        for fp in reversed(window):
            if len(fp.text) < _MIN_TEXT_LEN:
                continue
            # Exact match first
            if fp.rabin == rb and fp.text == text:
                duplicate_of = fp.index
                best_sim = 1.0
                break
            # Approximate near-duplicate via SimHash
            sim = hamming_similarity(sh, fp.simhash_val)
            if sim >= similarity_threshold and sim > best_sim:
                best_sim = sim
                duplicate_of = fp.index

        if duplicate_of is not None:
            pointer = f"[DEDUP:ref={duplicate_of} sim={best_sim:.2f}]"
            saved = max(0, len(text) - len(pointer))
            tokens_saved += int(saved / 3.5)
            new_msg = dict(msg)
            new_msg["content"] = pointer
            out_messages.append(new_msg)
        else:
            out_messages.append(msg)

        fps.append(_TurnFingerprint(i, text, rb, sh))

    return out_messages, tokens_saved
