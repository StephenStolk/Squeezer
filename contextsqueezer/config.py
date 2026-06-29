"""
Global configuration for ContextSqueezer.

Settings are read (in priority order) from:
  1. Environment variables prefixed with SQUEEZER_
  2. ~/.config/contextsqueezer/config.toml
  3. Built-in defaults below
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path.home() / ".config" / "contextsqueezer"
DB_PATH = CONFIG_DIR / "store.db"
LOG_PATH = CONFIG_DIR / "squeezer.log"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SQUEEZER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Proxy ────────────────────────────────────────────────────────────────
    proxy_host: str = Field("127.0.0.1", description="Interface to bind the proxy on")
    proxy_port: int = Field(8787, description="TCP port for the local loopback proxy")

    # ── Upstream providers ────────────────────────────────────────────────────
    anthropic_upstream: str = Field(
        "https://api.anthropic.com", description="Real Anthropic API base URL"
    )
    openai_upstream: str = Field(
        "https://api.openai.com", description="Real OpenAI API base URL"
    )
    openrouter_upstream: str = Field(
        "https://openrouter.ai/api", description="Real OpenRouter API base URL"
    )

    # ── Pipeline toggles ──────────────────────────────────────────────────────
    enable_ast_compactor: bool = True
    enable_call_graph_pruner: bool = True
    enable_json_crusher: bool = True
    enable_lsh_deduplicator: bool = True
    enable_shell_sandbox: bool = True
    enable_linguistic_minifier: bool = True
    enable_temporal_decay: bool = True
    enable_pii_scrubber: bool = True
    enable_cache_aligner: bool = True
    enable_ccr: bool = True
    enable_file_version_tracker: bool = True
    enable_component_router: bool = True

    # ── Aggressiveness knobs ──────────────────────────────────────────────────
    ast_strip_bodies: bool = Field(
        True, description="Strip function bodies from non-focal files"
    )
    lsh_similarity_threshold: float = Field(
        0.85, ge=0.0, le=1.0, description="SimHash similarity above which a chunk is a dupe"
    )
    temporal_recent_turns: int = Field(
        2, ge=1, description="Number of most-recent turns kept verbatim"
    )
    temporal_partial_turns: int = Field(
        8, ge=1, description="Turns kept with markdown stripped (after recent window)"
    )
    ccr_token_threshold: int = Field(
        2000, ge=100, description="Token estimate above which a block is offloaded to CCR"
    )
    session_fill_pct: float = Field(
        0.70, ge=0.1, le=0.99, description="Context-fill percentage that triggers auto-summarisation"
    )
    json_max_depth: int = Field(
        4, ge=1, description="Maximum nesting depth retained in JSON crusher output"
    )
    file_version_diff_threshold: float = Field(
        0.6, ge=0.05, le=1.0,
        description="A file diff is used only if smaller than this fraction of the new file's size",
    )

    # ── Storage ───────────────────────────────────────────────────────────────
    db_path: Path = Field(DB_PATH, description="SQLite database file path")

    # ── Dashboard ─────────────────────────────────────────────────────────────
    dashboard_port: int = Field(8788, description="TCP port for the local analytics dashboard")
    dashboard_enabled: bool = True

    # ── Traffic recording (for offline eval against real sessions) ────────────
    enable_recording: bool = Field(
        False, description="Record raw, pre-compression request payloads to a JSONL file"
    )
    recording_path: Path = Field(
        CONFIG_DIR / "recordings" / "raw_requests.jsonl",
        description="Where recorded raw payloads are appended, one JSON object per line",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_path: Path = Field(LOG_PATH, description="Path for structured log output")

    @field_validator("db_path", "log_path", "recording_path", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser()

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def proxy_base_url(self) -> str:
        return f"http://{self.proxy_host}:{self.proxy_port}"

    def ensure_dirs(self) -> None:
        """Create config / storage directories if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.recording_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
