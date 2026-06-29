"""
Cross-File Symbol Call-Graph Pruner

Maintains an offline SQLite index of the active project repository workspace.
Evaluates the active execution context (focal symbols) and automatically slices
out code bodies across background files unless they sit directly within the
transitive import or call pathway from the focal set.

Usage
-----
  indexer = CallGraphIndexer(db_path)
  await indexer.index_project("/path/to/project")

  pruner = CallGraphPruner(db_path)
  # symbols the agent is actively working with:
  pruned = await pruner.prune_file(code, "path/to/file.py", focal_symbols={"MyClass", "parse_payload"})
"""

from __future__ import annotations

import ast
import os
import re
import sqlite3
from pathlib import Path
from typing import Iterable


# ──────────────────────────────────────────────────────────────────────────────
# SQLite schema
# ──────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cg_symbols (
    file    TEXT NOT NULL,
    symbol  TEXT NOT NULL,
    kind    TEXT NOT NULL DEFAULT 'function',
    PRIMARY KEY (file, symbol)
);
CREATE TABLE IF NOT EXISTS cg_calls (
    caller_file     TEXT NOT NULL,
    caller_symbol   TEXT NOT NULL,
    callee_symbol   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cg_caller ON cg_calls(caller_file, caller_symbol);
CREATE INDEX IF NOT EXISTS idx_cg_callee ON cg_calls(callee_symbol);
"""


# ──────────────────────────────────────────────────────────────────────────────
# Python call-graph extractor (stdlib ast, no external deps)
# ──────────────────────────────────────────────────────────────────────────────

class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.symbols: list[tuple[str, str]] = []   # (name, kind)
        self.calls: list[tuple[str, str]] = []      # (from_symbol, to_symbol)
        self._current: str = "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.symbols.append((node.name, "function"))
        prev = self._current
        self._current = node.name
        self.generic_visit(node)
        self._current = prev

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbols.append((node.name, "class"))
        prev = self._current
        self._current = node.name
        self.generic_visit(node)
        self._current = prev

    def visit_Call(self, node: ast.Call) -> None:
        callee = ""
        if isinstance(node.func, ast.Name):
            callee = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee = node.func.attr
        if callee:
            self.calls.append((self._current, callee))
        self.generic_visit(node)


def _extract_python_graph(code: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (symbols, calls) extracted via Python AST."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [], []
    v = _Visitor()
    v.visit(tree)
    return v.symbols, v.calls


# ──────────────────────────────────────────────────────────────────────────────
# Indexer
# ──────────────────────────────────────────────────────────────────────────────

_PY_EXT = {".py", ".pyi"}
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", "dist", "build"}


class CallGraphIndexer:
    """Walks a project directory and builds a call-graph index in SQLite."""

    def __init__(self, db_path: Path) -> None:
        self._db = str(db_path)

    def _init_db(self, conn: sqlite3.Connection) -> None:
        conn.executescript(_SCHEMA)
        conn.commit()

    def index_project(self, project_root: str | Path) -> int:
        """Synchronously walk and index the project. Returns file count."""
        project_root = Path(project_root)
        count = 0
        with sqlite3.connect(self._db) as conn:
            self._init_db(conn)
            for root, dirs, files in os.walk(project_root):
                # Prune unwanted dirs in-place
                dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
                for fname in files:
                    fpath = Path(root) / fname
                    if fpath.suffix not in _PY_EXT:
                        continue
                    rel = str(fpath.relative_to(project_root))
                    try:
                        code = fpath.read_text(errors="ignore")
                        symbols, calls = _extract_python_graph(code)
                        # Clear existing entries for this file
                        conn.execute("DELETE FROM cg_symbols WHERE file=?", (rel,))
                        conn.execute("DELETE FROM cg_calls WHERE caller_file=?", (rel,))
                        conn.executemany(
                            "INSERT OR REPLACE INTO cg_symbols(file, symbol, kind) VALUES(?,?,?)",
                            [(rel, s, k) for s, k in symbols],
                        )
                        conn.executemany(
                            "INSERT INTO cg_calls(caller_file, caller_symbol, callee_symbol) "
                            "VALUES(?,?,?)",
                            [(rel, c, e) for c, e in calls],
                        )
                        count += 1
                    except (OSError, UnicodeDecodeError):
                        continue
            conn.commit()
        return count


# ──────────────────────────────────────────────────────────────────────────────
# Pruner
# ──────────────────────────────────────────────────────────────────────────────

class CallGraphPruner:
    """
    Given a set of focal symbols, decides whether a file's symbols are
    transitively reachable. For unreachable symbols, strips the body.
    """

    def __init__(self, db_path: Path) -> None:
        self._db = str(db_path)

    def _reachable_symbols(self, focal_symbols: Iterable[str]) -> set[str]:
        """BFS over call graph to find all symbols reachable from focal set."""
        reachable: set[str] = set(focal_symbols)
        queue = list(reachable)

        try:
            with sqlite3.connect(self._db) as conn:
                while queue:
                    sym = queue.pop()
                    rows = conn.execute(
                        "SELECT callee_symbol FROM cg_calls WHERE caller_symbol=?", (sym,)
                    ).fetchall()
                    for (callee,) in rows:
                        if callee not in reachable:
                            reachable.add(callee)
                            queue.append(callee)
        except sqlite3.OperationalError:
            pass  # DB not initialised – return focal set only

        return reachable

    def prune_file(
        self,
        code: str,
        file_path: str = "",
        focal_symbols: Iterable[str] | None = None,
    ) -> tuple[str, int]:
        """
        Return (pruned_code, tokens_saved).

        Symbols not in the reachable set from *focal_symbols* have their
        bodies replaced with `...`.  If *focal_symbols* is empty/None, no
        pruning occurs.
        """
        if not focal_symbols:
            return code, 0

        reachable = self._reachable_symbols(focal_symbols)

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, 0

        lines = code.splitlines(keepends=True)
        prune_ranges: list[tuple[int, int]] = []  # (start_line, end_line) 0-indexed

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name not in reachable:
                    # Find the body start (line after the def/class line)
                    body_start = node.body[0].lineno - 1
                    body_end = node.end_lineno  # type: ignore[attr-defined]
                    prune_ranges.append((body_start, body_end))

        if not prune_ranges:
            return code, 0

        # Sort and merge overlapping ranges
        prune_ranges.sort()
        merged: list[tuple[int, int]] = []
        for start, end in prune_ranges:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        original_len = len(code)
        out: list[str] = []
        prev_end = 0

        for start, end in merged:
            out.extend(lines[prev_end:start])
            # Insert placeholder indented to match body
            indent = len(lines[start]) - len(lines[start].lstrip()) if start < len(lines) else 4
            out.append(" " * indent + "...\n")
            prev_end = end

        out.extend(lines[prev_end:])
        pruned = "".join(out)
        saved = max(0, original_len - len(pruned))
        return pruned, int(saved / 3.5)
