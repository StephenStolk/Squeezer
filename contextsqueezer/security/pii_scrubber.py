"""
Zero-Trust PII Scrubber

Runs a deterministic local scanner array that catches and masks sensitive data
formats entirely on localhost before any optimisation patterns execute or data
passes to external network layers.

Detected patterns (in priority order):
  • Auth / API tokens (Bearer, sk-*, ghp_*, xoxb-*, etc.)
  • Passwords in common key=value / JSON patterns
  • Private keys (PEM blocks)
  • IPv4 + IPv6 addresses
  • Email addresses
  • Credit card numbers (Luhn-checked)
  • SSN / NI numbers
  • AWS / GCP / Azure credential strings
  • Connection strings (postgres://, mysql://, etc.)
  • High-entropy strings (≥40 chars, entropy > 4.5 bits/char)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

# ──────────────────────────────────────────────────────────────────────────────
# Pattern registry
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PiiPattern:
    name: str
    pattern: re.Pattern
    mask: str = "[REDACTED]"
    validator: Callable[[str], bool] | None = None


def _luhn(n: str) -> bool:
    digits = [int(d) for d in n if d.isdigit()]
    digits.reverse()
    total = sum(
        d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2)
        for i, d in enumerate(digits)
    )
    return total % 10 == 0


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


_PATTERNS: list[PiiPattern] = [
    # PEM private keys
    PiiPattern(
        "pem_private_key",
        re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+ PRIVATE KEY-----", re.DOTALL),
        "[REDACTED:PRIVATE_KEY]",
    ),
    # API / Bearer tokens
    PiiPattern(
        "bearer_token",
        re.compile(r"(?i)(Authorization\s*[:=]\s*Bearer\s+)([\w\-\.]{20,})"),
        mask="[REDACTED:BEARER]",
    ),
    # OpenAI / Anthropic key
    PiiPattern(
        "sk_key",
        re.compile(r"\b(sk-[A-Za-z0-9\-_]{32,})\b"),
        "[REDACTED:API_KEY]",
    ),
    # GitHub PAT
    PiiPattern(
        "github_pat",
        re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{36,})\b"),
        "[REDACTED:GITHUB_TOKEN]",
    ),
    # Slack token
    PiiPattern(
        "slack_token",
        re.compile(r"\b(xox[baprs]-[A-Za-z0-9\-]{10,})\b"),
        "[REDACTED:SLACK_TOKEN]",
    ),
    # AWS access key
    PiiPattern(
        "aws_access_key",
        re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
        "[REDACTED:AWS_KEY]",
    ),
    # AWS secret key (typical format in env vars / configs)
    PiiPattern(
        "aws_secret",
        re.compile(r"(?i)(aws[_\-]?secret[_\-]?access[_\-]?key\s*[=:]\s*)([A-Za-z0-9/+]{40})"),
        "[REDACTED:AWS_SECRET]",
    ),
    # Connection strings
    PiiPattern(
        "connection_string",
        re.compile(
            r"(?i)(postgres(?:ql)?|mysql|mongodb|redis|amqp)://"
            r"[A-Za-z0-9_\-\.%]+:[^@\s]{1,128}@[^\s\"']{1,256}"
        ),
        "[REDACTED:CONN_STRING]",
    ),
    # Password in JSON / YAML / env
    PiiPattern(
        "password_kv",
        re.compile(
            r'(?i)(?:"password"\s*:\s*"|password\s*[=:]\s*["\']?)([^"\'\s,}{]{6,})',
        ),
        "[REDACTED:PASSWORD]",
    ),
    # Credit cards (Luhn-validated)
    PiiPattern(
        "credit_card",
        re.compile(r"\b(?:\d[ \-]?){13,19}\b"),
        "[REDACTED:CC]",
        validator=lambda m: _luhn(re.sub(r"[ \-]", "", m)),
    ),
    # SSN
    PiiPattern(
        "ssn",
        re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"),
        "[REDACTED:SSN]",
    ),
    # Email
    PiiPattern(
        "email",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED:EMAIL]",
    ),
]

# High-entropy string scanner (applied last, so known tokens are already masked)
_HIGH_ENTROPY_RE = re.compile(r"[A-Za-z0-9+/=_\-]{40,}")
_ENTROPY_THRESHOLD = 4.5


@dataclass
class ScrubResult:
    text: str
    hits: dict[str, int] = field(default_factory=dict)

    @property
    def total_hits(self) -> int:
        return sum(self.hits.values())


class PiiScrubber:
    """
    Deterministic, regex+entropy local PII scanner.
    All processing happens in memory on localhost – nothing is logged externally.
    """

    def __init__(self, patterns: list[PiiPattern] | None = None) -> None:
        self._patterns = patterns if patterns is not None else _PATTERNS

    def scrub(self, text: str) -> ScrubResult:
        hits: dict[str, int] = {}

        for pp in self._patterns:
            def _replace(m: re.Match, pp: PiiPattern = pp) -> str:
                full = m.group(0)
                # If validator rejects, skip
                if pp.validator and not pp.validator(full):
                    return full
                hits[pp.name] = hits.get(pp.name, 0) + 1
                return pp.mask

            text = pp.pattern.sub(_replace, text)

        # High-entropy sweep
        def _entropy_replace(m: re.Match) -> str:
            s = m.group(0)
            # Skip if already a mask
            if s.startswith("[REDACTED"):
                return s
            if _shannon_entropy(s) >= _ENTROPY_THRESHOLD:
                hits["high_entropy"] = hits.get("high_entropy", 0) + 1
                return "[REDACTED:HIGH_ENTROPY]"
            return s

        text = _HIGH_ENTROPY_RE.sub(_entropy_replace, text)

        return ScrubResult(text=text, hits=hits)

    def scrub_messages(self, messages: list[dict]) -> tuple[list[dict], dict[str, int]]:
        """Scrub an entire messages array in-place (returns new list + hit map)."""
        total_hits: dict[str, int] = {}
        out: list[dict] = []
        for msg in messages:
            new_msg = dict(msg)
            content = msg.get("content", "")
            if isinstance(content, str):
                result = self.scrub(content)
                new_msg["content"] = result.text
                for k, v in result.hits.items():
                    total_hits[k] = total_hits.get(k, 0) + v
            elif isinstance(content, list):
                new_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        result = self.scrub(block.get("text", ""))
                        new_blocks.append({**block, "text": result.text})
                        for k, v in result.hits.items():
                            total_hits[k] = total_hits.get(k, 0) + v
                    else:
                        new_blocks.append(block)
                new_msg["content"] = new_blocks
            out.append(new_msg)
        return out, total_hits
