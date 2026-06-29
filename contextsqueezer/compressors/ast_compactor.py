"""
AST Syntax-Directed Compaction Engine

Uses Tree-sitter to parse source code and produce a skeleton representation:
  • Class declarations and function/method signatures are preserved verbatim.
  • Function bodies (for non-focal files) are replaced with `...` placeholders.
  • Docstrings and inline comments are stripped in aggressive mode.
  • Import blocks are preserved (critical for call-graph reasoning).

Focal files (the file the agent is actively editing) are kept verbatim.
"""

from __future__ import annotations

import re
from pathlib import Path

# Lazy import – tree-sitter may not be installed in all envs
try:
    from tree_sitter_languages import get_language, get_parser  # type: ignore

    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Language detection
# ──────────────────────────────────────────────────────────────────────────────

_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".swift": "swift",
}


def _detect_language(path: str | Path) -> str | None:
    ext = Path(path).suffix.lower()
    return _EXT_TO_LANG.get(ext)


# ──────────────────────────────────────────────────────────────────────────────
# Node type sets per language
# ──────────────────────────────────────────────────────────────────────────────

# Nodes whose *bodies* we strip (keeping signature line)
_BODY_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "async_function_definition"},
    "javascript": {"function_declaration", "method_definition", "arrow_function"},
    "typescript": {"function_declaration", "method_definition", "arrow_function"},
    "tsx": {"function_declaration", "method_definition", "arrow_function"},
    "rust": {"function_item"},
    "go": {"function_declaration", "method_declaration"},
    "java": {"method_declaration", "constructor_declaration"},
    "cpp": {"function_definition"},
    "c": {"function_definition"},
    "c_sharp": {"method_declaration", "constructor_declaration"},
    "ruby": {"method", "singleton_method"},
    "kotlin": {"function_declaration"},
    "swift": {"function_declaration"},
}

# Block body node names (the child that holds the body)
_BODY_CHILD_TYPES: set[str] = {
    "block",
    "statement_block",
    "compound_statement",
    "body",
    "function_body",
}


# ──────────────────────────────────────────────────────────────────────────────
# Pure-Python fallback (regex-based skeleton)
# ──────────────────────────────────────────────────────────────────────────────

_PY_DEF_RE = re.compile(
    r"^( {0,16}(?:async\s+)?def\s+\w+\([^)]*\)(?:\s*->[^:]+)?:)(.*)",
    re.MULTILINE,
)
_PY_COMMENT_RE = re.compile(r"#[^\n]*")
_PY_DOCSTRING_RE = re.compile(r'"""[\s\S]+?"""|\'\'\'[\s\S]+?\'\'\'', re.DOTALL)


def _python_skeleton_fallback(code: str, strip_docstrings: bool = True) -> str:
    """Best-effort Python skeleton without tree-sitter."""
    if strip_docstrings:
        code = _PY_DOCSTRING_RE.sub("", code)
        code = _PY_COMMENT_RE.sub("", code)

    lines = code.splitlines(keepends=True)
    out: list[str] = []
    inside_body = False
    body_indent: int | None = None

    for line in lines:
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)

        if inside_body:
            if not stripped or current_indent > body_indent:  # type: ignore[operator]
                continue  # still inside body – skip
            else:
                inside_body = False  # body ended

        m = _PY_DEF_RE.match(line)
        if m:
            out.append(m.group(1) + "\n")
            out.append(" " * (current_indent + 4) + "...\n")
            inside_body = True
            body_indent = current_indent
        else:
            out.append(line)

    return "".join(out)


# ──────────────────────────────────────────────────────────────────────────────
# Tree-sitter compactor
# ──────────────────────────────────────────────────────────────────────────────

def _ts_skeleton(code: str, language: str) -> str:
    """Use tree-sitter to produce a skeleton of the source code."""
    try:
        parser = get_parser(language)
        tree = parser.parse(code.encode())
    except Exception:
        return code  # graceful degradation

    body_types = _BODY_NODE_TYPES.get(language, set())
    code_bytes = code.encode()
    result: list[str] = []
    cursor = 0  # byte offset we've emitted up to

    def _walk(node) -> None:  # type: ignore[no-untyped-def]
        nonlocal cursor

        if node.type in body_types:
            # Emit signature (up to first child that is a body block)
            body_child = None
            for child in node.children:
                if child.type in _BODY_CHILD_TYPES:
                    body_child = child
                    break

            if body_child:
                # Emit everything from cursor to start of body
                result.append(code_bytes[cursor : body_child.start_byte].decode(errors="replace"))
                # Emit placeholder body
                result.append("{ ... }")
                cursor = body_child.end_byte
                return  # don't recurse into body

        # Default: recurse into children
        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    # Emit any remaining code after last processed node
    result.append(code_bytes[cursor:].decode(errors="replace"))
    return "".join(result)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def compact_code(
    code: str,
    *,
    file_path: str | Path = "",
    strip_comments: bool = True,
    strip_docstrings: bool = True,
) -> tuple[str, int]:
    """
    Return (compacted_code, tokens_saved_estimate).

    Falls back to regex-based Python skeleton if tree-sitter is unavailable
    or the language is not supported.
    """
    original_len = len(code)

    language = _detect_language(file_path) if file_path else None

    if _TS_AVAILABLE and language:
        try:
            compacted = _ts_skeleton(code, language)
        except Exception:
            compacted = _python_skeleton_fallback(code, strip_docstrings)
    elif language == "python" or not language:
        compacted = _python_skeleton_fallback(code, strip_docstrings)
    else:
        # Unknown language – strip comments only
        compacted = code

    if strip_comments and language in (None, "python"):
        compacted = _PY_COMMENT_RE.sub("", compacted)

    saved_chars = max(0, original_len - len(compacted))
    saved_tokens = int(saved_chars / 3.5)
    return compacted, saved_tokens
