from contextsqueezer.security.pii_scrubber import PiiScrubber


def test_scrub_api_key():
    scrubber = PiiScrubber()
    text = "Here is my key: sk-abcdefghijklmnopqrstuvwxyz1234567890ABCD"
    result = scrubber.scrub(text)
    assert "sk-abc" not in result.text
    assert "API_KEY" in result.text


def test_scrub_email():
    scrubber = PiiScrubber()
    text = "Contact me at john.doe@example.com for details."
    result = scrubber.scrub(text)
    assert "john.doe@example.com" not in result.text
    assert "EMAIL" in result.text


def test_scrub_github_token():
    scrubber = PiiScrubber()
    text = "token: ghp_1234567890abcdefghijklmnopqrstuvwx12"
    result = scrubber.scrub(text)
    assert "ghp_1234567890" not in result.text


def test_scrub_no_false_positive_on_normal_text():
    scrubber = PiiScrubber()
    text = "The quick brown fox jumps over the lazy dog."
    result = scrubber.scrub(text)
    assert result.text == text
    assert result.total_hits == 0


def test_scrub_messages_list():
    scrubber = PiiScrubber()
    messages = [
        {"role": "user", "content": "my email is test@test.com"},
        {"role": "assistant", "content": "Got it."},
    ]
    out, hits = scrubber.scrub_messages(messages)
    assert "test@test.com" not in out[0]["content"]
    assert hits.get("email", 0) >= 1
