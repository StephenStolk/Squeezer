from contextsqueezer.compressors.ast_compactor import compact_code


PY_SAMPLE = '''
import os
import sys

class Foo:
    """A docstring."""

    def bar(self, x):
        """Bar docstring."""
        y = x + 1
        z = y * 2
        return z

    def baz(self):
        # a comment
        return 42


def standalone(a, b):
    total = 0
    for i in range(a):
        total += i * b
    return total
'''


def test_compact_code_strips_bodies():
    compacted, saved = compact_code(PY_SAMPLE, file_path="sample.py")
    assert "def bar" in compacted
    assert "def baz" in compacted
    assert "def standalone" in compacted
    # Bodies should be gone
    assert "total += i * b" not in compacted
    assert saved >= 0


def test_compact_code_handles_empty():
    compacted, saved = compact_code("", file_path="empty.py")
    assert compacted == ""
    assert saved == 0


def test_compact_code_unknown_language_passthrough():
    code = "some random text that is not code at all"
    compacted, saved = compact_code(code, file_path="file.unknown")
    assert isinstance(compacted, str)
