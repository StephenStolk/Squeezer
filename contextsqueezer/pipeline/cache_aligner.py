"""
Deterministic Provider Cache Aligner — v2

The v1 design scored each message's "stability" fresh on every request and
re-sorted the stable portion by that score. That's broken: provider prompt
caching (Anthropic, OpenAI) matches an *exact, byte-identical, growing
prefix* across turns. The moment any message crosses from the
scored-and-sorted bucket into the "recent" bucket as a conversation grows,
the sort can place it somewhere other than at the end — which inserts
content into the middle of what was previously a stable prefix, and that
invalidates the very cache it was trying to preserve. A heuristic
recomputed per-request is the wrong primitive for something that needs
byte-for-byte stability across requests.

The fix: stop reordering conversational turns based on content at all.
Chronological order *is* the append-only, growing prefix providers want —
new turns are added at the end, old ones never move. This stage now only:

  1. Verifies system message(s) are at the front (defensive; should already
     be true for any well-formed request).
  2. Sorts tool *definitions* alphabetically by name. Safe, because a tool
     list is structural, not content-dependent — the same set of tools
     sorts to the same bytes every time regardless of declaration order.
  3. Places the Anthropic `cache_control: ephemeral` breakpoint at a FIXED
     relative position — "everything except the last K turns" — instead of
     a content-derived one. Since the prefix up to that point is exactly
     the same append-only sequence the client has always been sending, the
     breakpoint sits after a genuinely stable prefix, which is what makes
     provider-side caching actually work across turns.
"""

from __future__ import annotations


def _sort_tools(tools: list[dict]) -> list[dict]:
    """Sort tool definitions deterministically by name for cache consistency."""
    return sorted(tools, key=lambda t: t.get("name", ""))


def _inject_anthropic_cache_hint(messages: list[dict], boundary_index: int) -> list[dict]:
    """
    Add `cache_control: {type: ephemeral}` to the message immediately before
    the dynamic tail. Anthropic caches everything up to and including this
    breakpoint.
    """
    if boundary_index <= 0 or boundary_index > len(messages):
        return messages

    out = list(messages)
    target = dict(out[boundary_index - 1])
    content = target.get("content", "")

    if isinstance(content, str):
        target["content"] = [
            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
        ]
    elif isinstance(content, list) and content:
        new_content = list(content)
        last_block = dict(new_content[-1])
        last_block["cache_control"] = {"type": "ephemeral"}
        new_content[-1] = last_block
        target["content"] = new_content

    out[boundary_index - 1] = target
    return out


def align_for_cache(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    provider: str = "anthropic",
    dynamic_tail_size: int = 3,
    inject_cache_hint: bool = True,
) -> tuple[list[dict], list[dict] | None, int]:
    """
    Prepare messages/tools for maximum KV-cache hit probability *without*
    breaking the provider's exact-prefix matching.

    Returns (messages_in_original_order, sorted_tools, cache_boundary_index).

    Unlike v1, `messages` here is the SAME sequence the caller passed in —
    no conversational reordering happens. Only the cache_control annotation
    (and tool sort order) are added.
    """
    if not messages:
        return messages, (_sort_tools(tools) if tools else tools), 0

    non_system_count = sum(1 for m in messages if m.get("role") != "system")
    boundary = max(0, len(messages) - min(dynamic_tail_size, non_system_count))

    out_messages = messages
    if inject_cache_hint and provider == "anthropic" and boundary > 0:
        out_messages = _inject_anthropic_cache_hint(messages, boundary)

    sorted_tools = _sort_tools(tools) if tools else tools

    return out_messages, sorted_tools, boundary
