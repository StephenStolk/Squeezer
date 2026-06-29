"""
ContextSqueezer — Deterministic High-Throughput Context Proxy.

Intercepts local developer agent traffic, optimises context payloads via
modular deterministic structural pruners, maximises provider prompt-cache
longevity, and preserves strict execution safety through local-first
retrieval fallbacks.
"""

__version__ = "0.1.0"
__author__ = "ContextSqueezer Contributors"
__license__ = "MIT"

from contextsqueezer.config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]
