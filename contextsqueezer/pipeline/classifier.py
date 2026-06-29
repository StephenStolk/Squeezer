"""
Payload Structural Classifier

Separates the inbound message stream into specialised categories so that
the appropriate reduction engines can be dispatched in parallel:

  CODE        – Source code blocks (detected by fence, extension, or AST parse)
  JSON_DATA   – JSON objects / arrays (configs, tool schemas, log dumps)
  SHELL_OUTPUT – Terminal stdout/stderr (stack traces, test output, CLI logs)
  CONVERSATION – Prose conversational turns
  MIXED        – Content that spans multiple categories
"""

from __future__ import annotations

import json
import re
from enum import Enum, auto
from typing import NamedTuple


class ContentKind(Enum):
    CODE = auto()
    JSON_DATA = auto()
    SHELL_OUTPUT = auto()
    CONVERSATION = auto()
    MIXED = auto()


class ClassifiedBlock(NamedTuple):
    kind: ContentKind
    text: str
    language: str | None = None   # set for CODE blocks


# ──────────────────────────────────────────────────────────────────────────────
# Detection heuristics
# ──────────────────────────────────────────────────────────────────────────────

_CODE_FENCE_RE = re.compile(r"^```(\w+)?", re.MULTILINE)
_FILE_PATH_RE = re.compile(r"(?:^|\s)([\w\-./]+\.[a-z]{2,6}):", re.MULTILINE)
_SHELL_SIGNAL_RE = re.compile(
    r"(Traceback \(most recent call last\)|Error:|Exception:|FAILED|PASSED|"
    r"\$ |\[\d+\]|>>>\s|#\s+|→|✓|✗|pytest|cargo test|go test)",
    re.MULTILINE,
)
_JSON_START_RE = re.compile(r"^\s*[\[{]")

_CODE_KEYWORDS = frozenset(
    ["def ", "class ", "function ", "const ", "let ", "var ", "import ", "from ",
     "public ", "private ", "#include", "fn ", "func ", "package "]
)


def _looks_like_code(text: str) -> bool:
    if _CODE_FENCE_RE.search(text):
        return True
    hits = sum(1 for kw in _CODE_KEYWORDS if kw in text)
    return hits >= 2


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if not _JSON_START_RE.match(stripped):
        return False
    try:
        json.loads(stripped)
        return True
    except json.JSONDecodeError:
        # Large JSON may be truncated – heuristic check
        return len(stripped) > 100 and stripped.startswith(("{", "["))


def _looks_like_shell(text: str) -> bool:
    return bool(_SHELL_SIGNAL_RE.search(text))


def classify_text(text: str) -> ContentKind:
    kinds: set[ContentKind] = set()
    if _looks_like_json(text):
        kinds.add(ContentKind.JSON_DATA)
    if _looks_like_code(text):
        kinds.add(ContentKind.CODE)
    if _looks_like_shell(text):
        kinds.add(ContentKind.SHELL_OUTPUT)

    if len(kinds) == 0:
        return ContentKind.CONVERSATION
    if len(kinds) == 1:
        return next(iter(kinds))
    return ContentKind.MIXED


# ──────────────────────────────────────────────────────────────────────────────
# Block splitter (splits fenced code blocks from surrounding prose)
# ──────────────────────────────────────────────────────────────────────────────

_FENCE_BLOCK_RE = re.compile(
    r"(```(\w+)?\n[\s\S]+?```)", re.MULTILINE
)

# Language tags that map directly to a kind without needing to sniff content.
_KNOWN_CODE_LANGS = frozenset(
    "python py javascript js typescript ts tsx jsx rust rs go golang java "
    "c cpp c++ csharp cs ruby rb php kotlin kt swift sql html css yaml yml "
    "toml xml".split()
)
_LANG_TAG_TO_KIND: dict[str, ContentKind] = {
    "json": ContentKind.JSON_DATA,
    "bash": ContentKind.SHELL_OUTPUT,
    "sh": ContentKind.SHELL_OUTPUT,
    "shell": ContentKind.SHELL_OUTPUT,
    "console": ContentKind.SHELL_OUTPUT,
    "output": ContentKind.SHELL_OUTPUT,
    "log": ContentKind.SHELL_OUTPUT,
}


def _fence_inner_text(full_block: str) -> str:
    """Strip the ```lang and closing ``` lines, leaving just the fenced content."""
    lines = full_block.split("\n")
    if len(lines) >= 2:
        return "\n".join(lines[1:-1])
    return full_block


def _classify_fence(full_block: str, lang: str | None) -> ContentKind:
    """
    A fenced fragment isn't necessarily code just because it's fenced — agents
    routinely wrap shell output, JSON dumps, or plain text in ``` with no (or a
    misleading) language tag. Trust an unambiguous tag; otherwise sniff the
    actual inner content instead of defaulting to CODE.
    """
    if lang:
        lang_lower = lang.lower()
        if lang_lower in _KNOWN_CODE_LANGS:
            return ContentKind.CODE
        if lang_lower in _LANG_TAG_TO_KIND:
            return _LANG_TAG_TO_KIND[lang_lower]

    inner = _fence_inner_text(full_block)
    if not inner.strip():
        return ContentKind.CODE
    return classify_text(inner)


def split_into_blocks(text: str) -> list[ClassifiedBlock]:
    """
    Split *text* into typed blocks.  Fenced blocks are extracted first and
    classified by language tag (when unambiguous) or by sniffing their inner
    content; everything else is classified as prose/JSON/shell.
    """
    blocks: list[ClassifiedBlock] = []
    last_end = 0

    for m in _FENCE_BLOCK_RE.finditer(text):
        # Text before the code block
        before = text[last_end : m.start()]
        if before.strip():
            kind = classify_text(before)
            blocks.append(ClassifiedBlock(kind, before))

        # The fenced block itself — classify, don't assume.
        lang = m.group(2) or None
        fence_kind = _classify_fence(m.group(1), lang)
        blocks.append(ClassifiedBlock(fence_kind, m.group(1), language=lang))
        last_end = m.end()

    # Remaining text after last code block
    tail = text[last_end:]
    if tail.strip():
        kind = classify_text(tail)
        blocks.append(ClassifiedBlock(kind, tail))

    if not blocks:
        kind = classify_text(text)
        blocks.append(ClassifiedBlock(kind, text))

    return blocks


def classify_messages(messages: list[dict]) -> list[tuple[dict, ContentKind]]:
    """Return [(message, dominant_kind)] for each message in the list."""
    result = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        else:
            text = str(content)
        result.append((msg, classify_text(text)))
    return result
