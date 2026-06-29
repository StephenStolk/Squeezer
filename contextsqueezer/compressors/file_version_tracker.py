"""
File Version Tracker — diff-based delta encoding for repeatedly-read files.

Coding agents re-read the same file many times across a session, almost
always after making a small edit to it. Generic deduplication (LSH/SimHash)
only catches *identical* repeats — it has nothing useful to say about
"this file is 95% the same as last time, here's what changed."

This tracker keeps a per-file-path version chain in local SQLite. When a
file path reappears:
  • unchanged content        → tiny pointer, no content resent at all
  • changed, diff is cheap   → unified diff against the last version
  • changed, diff isn't cheap (near-total rewrite) → store as a fresh
    full version (a diff against a barely-related predecessor wastes
    more tokens than it saves)

This is a different, narrower mechanism than CCR: CCR offloads on raw size
regardless of history; this tracker offloads on *redundancy with a known
predecessor*, which is the dominant pattern in real agentic coding sessions.
"""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextsqueezer.storage.sqlite_store import Store


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class VersionResult:
    text: str            # what to actually send upstream in place of the file
    is_delta: bool        # True if `text` is a pointer or diff, not full content
    tokens_saved: int


class FileVersionTracker:
    """
    Per-request wrapper around the SQLite-backed file version chain.

    diff_threshold_ratio: a diff is only used if it's smaller than this
    fraction of the new file's full size. Above that, the file is treated
    as effectively rewritten and stored fresh instead.
    """

    def __init__(self, store: "Store", diff_threshold_ratio: float = 0.6) -> None:
        self._store = store
        self._threshold = diff_threshold_ratio

    async def process(self, file_path: str, content: str) -> VersionResult:
        if not file_path or not content.strip():
            return VersionResult(text=content, is_delta=False, tokens_saved=0)

        new_hash = _hash(content)
        prev = await self._store.file_version_get_latest(file_path)

        if prev is None:
            await self._store.file_version_put(file_path, new_hash, content, version=1)
            return VersionResult(text=content, is_delta=False, tokens_saved=0)

        if prev["content_hash"] == new_hash:
            pointer = f"[FILEREF:{file_path}@v{prev['version']} unchanged]"
            saved = max(0, len(content) - len(pointer))
            return VersionResult(text=pointer, is_delta=True, tokens_saved=int(saved / 3.5))

        diff_lines = list(
            difflib.unified_diff(
                prev["content"].splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"{file_path}@v{prev['version']}",
                tofile=f"{file_path}@v{prev['version'] + 1}",
                n=2,
            )
        )
        diff_text = "".join(diff_lines)
        new_version = prev["version"] + 1

        if diff_text and len(diff_text) < len(content) * self._threshold:
            await self._store.file_version_put(file_path, new_hash, content, version=new_version)
            wrapped = (
                f"[FILEDIFF:{file_path}@v{prev['version']}->v{new_version}]\n{diff_text}"
            )
            saved = max(0, len(content) - len(wrapped))
            return VersionResult(text=wrapped, is_delta=True, tokens_saved=int(saved / 3.5))

        # Diff isn't worth it — store and send the full new version.
        await self._store.file_version_put(file_path, new_hash, content, version=new_version)
        return VersionResult(text=content, is_delta=False, tokens_saved=0)
