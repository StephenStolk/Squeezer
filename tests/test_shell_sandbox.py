from contextsqueezer.compressors.shell_sandbox import minify_shell_output


PYTEST_OUTPUT = """
collecting ... 
test_foo.py::test_one PASSED
test_foo.py::test_two PASSED
test_foo.py::test_three FAILED

Traceback (most recent call last):
  File "/usr/lib/python3.11/site-packages/pytest/runner.py", line 100, in run
    result = func()
  File "/usr/lib/python3.11/site-packages/pytest/runner.py", line 102, in run
    result = func()
  File "test_foo.py", line 10, in test_three
    assert 1 == 2
AssertionError: assert 1 == 2

5 passed, 1 failed
"""


def test_minify_removes_passing_lines():
    result, saved = minify_shell_output(PYTEST_OUTPUT)
    assert "PASSED" not in result
    assert "AssertionError" in result


def test_minify_preserves_root_exception():
    result, saved = minify_shell_output(PYTEST_OUTPUT)
    assert "AssertionError" in result


def test_minify_collapses_stdlib_frames():
    result, saved = minify_shell_output(PYTEST_OUTPUT)
    assert "stdlib/venv frame" in result or "site-packages" not in result


def test_minify_handles_empty():
    result, saved = minify_shell_output("")
    assert result == ""
    assert saved == 0
