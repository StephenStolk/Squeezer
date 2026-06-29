import pytest
from pathlib import Path

from contextsqueezer.storage.sqlite_store import Store, init_db
from contextsqueezer.compressors.file_version_tracker import FileVersionTracker


@pytest.fixture
async def store(tmp_path: Path):
    db_path = tmp_path / "fv_test.db"
    await init_db(db_path)
    async with Store(db_path) as s:
        yield s


FILE_V1 = "\n".join(
    [f"def function_{i}(x):" for i in range(40)]
    + ["    return x + 1" for _ in range(40)]
) + "\ndef bar():\n    return 42\n"

FILE_V2 = FILE_V1.replace("return 42", "return 43  # changed")

FILE_V3_REWRITE = """class TotallyDifferent:
    def __init__(self):
        self.value = 1

    def compute(self):
        return self.value * 100
"""


async def test_first_version_stored_full(store):
    tracker = FileVersionTracker(store)
    result = await tracker.process("app.py", FILE_V1)
    assert result.text == FILE_V1
    assert result.is_delta is False


async def test_unchanged_file_becomes_pointer(store):
    tracker = FileVersionTracker(store)
    await tracker.process("app.py", FILE_V1)
    result = await tracker.process("app.py", FILE_V1)  # same content again
    assert result.is_delta is True
    assert "[FILEREF:" in result.text
    assert "unchanged" in result.text


async def test_small_edit_produces_diff(store):
    tracker = FileVersionTracker(store)
    await tracker.process("app.py", FILE_V1)
    result = await tracker.process("app.py", FILE_V2)
    assert result.is_delta is True
    assert "[FILEDIFF:" in result.text
    assert len(result.text) < len(FILE_V2)


async def test_total_rewrite_falls_back_to_full_content(store):
    tracker = FileVersionTracker(store, diff_threshold_ratio=0.3)
    await tracker.process("app.py", FILE_V1)
    result = await tracker.process("app.py", FILE_V3_REWRITE)
    # A near-total rewrite shouldn't be diffed — full content sent instead.
    assert result.text == FILE_V3_REWRITE
    assert result.is_delta is False


async def test_version_chain_persists_across_tracker_instances(store):
    tracker_a = FileVersionTracker(store)
    await tracker_a.process("shared.py", FILE_V1)

    tracker_b = FileVersionTracker(store)  # simulates a new request
    result = await tracker_b.process("shared.py", FILE_V1)
    assert result.is_delta is True  # tracker_b sees tracker_a's stored version


async def test_empty_content_passthrough(store):
    tracker = FileVersionTracker(store)
    result = await tracker.process("empty.py", "")
    assert result.text == ""
    assert result.tokens_saved == 0
