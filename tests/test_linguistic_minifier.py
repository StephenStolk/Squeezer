from contextsqueezer.compressors.linguistic_minifier import minify_text


def test_minify_strips_filler_openers():
    text = "Certainly! Here is the code you asked for."
    result, saved = minify_text(text)
    assert "Certainly" not in result


def test_minify_compacts_verbose_phrases():
    text = "I did this in order to make use of the available resources."
    result, saved = minify_text(text)
    assert "in order to" not in result
    assert "make use of" not in result


def test_minify_strips_trailing_helpfulness():
    text = "Here is the answer.\n\nLet me know if you have any other questions!"
    result, saved = minify_text(text)
    assert "Let me know" not in result


def test_minify_preserves_meaningful_content():
    text = "def calculate_total(items): return sum(items)"
    result, saved = minify_text(text)
    assert "calculate_total" in result
