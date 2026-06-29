"""
Content-Compressed Retrieval (CCR)

When a payload chunk exceeds `ccr_token_threshold`, CCR:
  1. Stores the full chunk in the local SQLite store under a SHA-256 hash.
  2. Replaces the chunk in the prompt with a one-line pointer token:
       [CCR:abc123de | label | ~2400 tok]
  3. Injects a `squeezer_retrieve` tool definition into the outgoing request
     so the upstream LLM can pull the raw content back if it needs it.

On a tool-call return path, the proxy intercepts `squeezer_retrieve` calls
and resolves them from SQLite before they ever hit the network.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextsqueezer.storage.sqlite_store import Store

CCR_TOOL_DEFINITION = {
    "name": "squeezer_retrieve",
    "description": (
        "Fetch the full raw content of a previously compressed context block. "
        "Use this when you need complete implementation details that were summarised "
        "by the ContextSqueezer proxy. Pass the hash shown in the [CCR:...] token."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "hash": {
                "type": "string",
                "description": "The hex hash ID from the [CCR:hash | ...] pointer token.",
            }
        },
        "required": ["hash"],
    },
}

# Matches: [CCR:abc123de | some label | ~2400 tok]
_CCR_POINTER_RE = re.compile(r"\[CCR:([0-9a-f]{16})\s*\|[^\]]+\]")

# Rough character-to-token ratio (conservative)
_CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def make_pointer(hash_id: str, label: str, token_estimate: int) -> str:
    return f"[CCR:{hash_id} | {label} | ~{token_estimate} tok]"


class CCRManager:
    def __init__(self, store: "Store", token_threshold: int = 2000) -> None:
        self._store = store
        self._threshold = token_threshold
        self._used = False  # did we offload anything in this request?

    @property
    def was_used(self) -> bool:
        return self._used

    def mark_used(self) -> None:
        """
        Allow other components (e.g. cross-component dedup) that store
        content via this same CCR-backed mechanism to register that the
        `squeezer_retrieve` tool must be injected this request too.
        """
        self._used = True

    async def force_offload(self, text: str, label: str = "") -> str:
        """
        Unconditionally store *text* in CCR and return a pointer, regardless
        of the token threshold. Used when something other than size makes
        the content worth offloading (e.g. it's a confirmed cross-component
        duplicate, so a pointer is strictly better than resending it).
        """
        hash_id = await self._store.ccr_put(text, label)
        self._used = True
        return make_pointer(hash_id, label or "chunk", estimate_tokens(text))

    async def maybe_offload(self, text: str, label: str = "") -> str:
        """
        If *text* exceeds the token threshold, store it and return a pointer.
        Otherwise return the text unchanged.
        """
        if estimate_tokens(text) < self._threshold:
            return text
        return await self.force_offload(text, label)

    async def resolve_pointer(self, pointer_text: str) -> str | None:
        """
        If *pointer_text* is a CCR pointer, fetch and return the original.
        Returns None if the text is not a CCR pointer.
        """
        m = _CCR_POINTER_RE.fullmatch(pointer_text.strip())
        if not m:
            return None
        return await self._store.ccr_get(m.group(1))

    async def handle_tool_call(self, tool_input: dict) -> str:
        """Resolve a `squeezer_retrieve` tool call from the upstream LLM."""
        hash_id = tool_input.get("hash", "")
        result = await self._store.ccr_get(hash_id)
        if result is None:
            return f"[CCR ERROR] No entry found for hash '{hash_id}'."
        return result

    def inject_tool(self, tools: list) -> list:
        """
        Append the squeezer_retrieve tool definition if CCR is active.
        Idempotent – won't add duplicates.
        """
        names = {t.get("name") for t in tools if isinstance(t, dict)}
        if "squeezer_retrieve" not in names:
            tools = list(tools) + [CCR_TOOL_DEFINITION]
        return tools
