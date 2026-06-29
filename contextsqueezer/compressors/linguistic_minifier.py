"""
Linguistic Minification Matrix

Aggressively minifies conversational / prose content by:
  1. Removing filler phrases and padding (no meaning, pure tokens).
  2. Replacing common verbose engineering phrases with compact equivalents.
  3. Collapsing redundant whitespace and normalising line endings.
  4. Stripping markdown decoration from turns that don't need it.

This module operates purely on string content and is fully reversible
conceptually (the mapping is a dictionary, though we don't expand on output
– the upstream LLM receives the compact form directly).
"""

from __future__ import annotations

import re

# ──────────────────────────────────────────────────────────────────────────────
# Filler patterns to strip entirely
# ──────────────────────────────────────────────────────────────────────────────

_FILLER_PATTERNS: list[re.Pattern] = [
    # Sycophantic openers
    re.compile(
        r"(?i)^\s*(certainly|absolutely|of course|sure thing|great question"
        r"|great|excellent|perfect|awesome|no problem|gladly)[,!.]?\s*",
        re.MULTILINE,
    ),
    # "I'll/I will/Let me" preambles
    re.compile(
        r"(?i)(I(?:'ll| will| can| would be happy to))\s+(?:help you\s+)?(with that|do that|try to|go ahead and|now)\s*",
        re.MULTILINE,
    ),
    # Meta-narration ("Let me now explain...", "I'll start by...", "First, I'll...")
    re.compile(
        r"(?i)(let(?:'s| me| us))\s+(now\s+)?(?:take a look|look at|start|begin|walk you through|explain|break this down|dive in|go over)",
        re.MULTILINE,
    ),
    # "As an AI language model..."
    re.compile(r"(?i)as an (AI|artificial intelligence) (language model|assistant)[,.]?\s*"),
    # "It's worth noting that"
    re.compile(r"(?i)it'?s worth (noting|mentioning|pointing out) (that\s*)?"),
    # "Please note that"
    re.compile(r"(?i)please note (that\s*)?"),
    # "In summary / To summarise / In conclusion" at line start (already implied)
    re.compile(r"(?i)^(in summary|to summarize|to summarise|in conclusion)[,:\s]+", re.MULTILINE),
    # Trailing "I hope this helps!" / "Let me know if..."
    re.compile(
        r"(?i)(I hope (this|that) helps?[!.]?|let me know if you (have|need|want)[^.]*\.?|feel free to ask[^.]*\.?)\s*$",
        re.MULTILINE,
    ),
    # Consecutive blank lines → single blank line
    re.compile(r"\n{3,}"),
]

_FILLER_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\n{3,}"), "\n\n"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Verbose → compact phrase dictionary
# ──────────────────────────────────────────────────────────────────────────────

_VERBOSE_TO_COMPACT: list[tuple[re.Pattern, str]] = [
    # "in order to" → "to"
    (re.compile(r"\bin order to\b", re.IGNORECASE), "to"),
    # "as well as" → "and"
    (re.compile(r"\bas well as\b", re.IGNORECASE), "and"),
    # "due to the fact that" → "because"
    (re.compile(r"\bdue to the fact that\b", re.IGNORECASE), "because"),
    # "at this point in time" → "now"
    (re.compile(r"\bat this point in time\b", re.IGNORECASE), "now"),
    # "in the event that" → "if"
    (re.compile(r"\bin the event that\b", re.IGNORECASE), "if"),
    # "is able to" → "can"
    (re.compile(r"\bis able to\b", re.IGNORECASE), "can"),
    # "are able to" → "can"
    (re.compile(r"\bare able to\b", re.IGNORECASE), "can"),
    # "make use of" → "use"
    (re.compile(r"\bmake use of\b", re.IGNORECASE), "use"),
    # "the reason why" → "why"
    (re.compile(r"\bthe reason why\b", re.IGNORECASE), "why"),
    # "whether or not" → "whether"
    (re.compile(r"\bwhether or not\b", re.IGNORECASE), "whether"),
    # "a number of" → "several"
    (re.compile(r"\ba number of\b", re.IGNORECASE), "several"),
    # "on a regular basis" → "regularly"
    (re.compile(r"\bon a regular basis\b", re.IGNORECASE), "regularly"),
    # "in the process of" → "while"
    (re.compile(r"\bin the process of\b", re.IGNORECASE), "while"),
    # "for the purpose of" → "for"
    (re.compile(r"\bfor the purpose of\b", re.IGNORECASE), "for"),
]

# ──────────────────────────────────────────────────────────────────────────────
# Markdown decoration stripping (for turns that don't need visual formatting)
# ──────────────────────────────────────────────────────────────────────────────

_MD_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*{2}([^*]+)\*{2}")
_MD_ITALIC_RE = re.compile(r"\*([^*]+)\*")
_MD_STRIKETHROUGH_RE = re.compile(r"~~([^~]+)~~")
_TRAILING_SPACES_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_MULTIPLE_SPACES_RE = re.compile(r"  +")


def strip_markdown_decoration(text: str) -> str:
    """Remove markdown heading markers and bold/italic decorators."""
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    text = _MD_STRIKETHROUGH_RE.sub(r"\1", text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def minify_text(
    text: str,
    *,
    strip_md: bool = False,
    strip_fillers: bool = True,
    compact_phrases: bool = True,
) -> tuple[str, int]:
    """
    Apply linguistic minification to *text*.

    Returns (minified_text, tokens_saved_estimate).
    """
    original_len = len(text)

    if strip_fillers:
        for pattern in _FILLER_PATTERNS:
            text = pattern.sub("", text)

    if compact_phrases:
        for pattern, replacement in _VERBOSE_TO_COMPACT:
            text = pattern.sub(replacement, text)

    if strip_md:
        text = strip_markdown_decoration(text)

    # Normalise whitespace
    for pattern, replacement in _FILLER_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = _TRAILING_SPACES_RE.sub("", text)
    text = _MULTIPLE_SPACES_RE.sub(" ", text)
    text = text.strip()

    saved = max(0, original_len - len(text))
    return text, int(saved / 3.5)


def minify_message(msg: dict, **kwargs) -> tuple[dict, int]:  # type: ignore[type-arg]
    """Minify the content of a single message dict. Returns (new_msg, tokens_saved)."""
    content = msg.get("content", "")
    total_saved = 0

    if isinstance(content, str):
        new_content, saved = minify_text(content, **kwargs)
        total_saved += saved
        return {**msg, "content": new_content}, total_saved

    if isinstance(content, list):
        new_blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                new_text, saved = minify_text(block.get("text", ""), **kwargs)
                total_saved += saved
                new_blocks.append({**block, "text": new_text})
            else:
                new_blocks.append(block)
        return {**msg, "content": new_blocks}, total_saved

    return msg, 0
