"""
ContextSqueezer Proxy Server

Starts a local HTTP reverse proxy on localhost:8787 (configurable).

Request lifecycle
-----------------
  1. Receive incoming request from local agent (Claude Code, Cursor, etc.)
  2. Detect provider from Host header or URL path prefix.
  3. (Optional) Record the raw payload to disk for later offline eval.
  4. Run the full compression pipeline (component-aware, budget-aware).
  5. Forward compressed payload to the real upstream provider.
  6. Watch the response for `squeezer_retrieve` tool calls and resolve them
     from SQLite before they ever leave the machine.
  7. Record metrics (including per-component breakdown) asynchronously.

Provider routing
----------------
  /v1/messages          → Anthropic
  /openai/v1/chat/...  → OpenAI
  /openrouter/api/...  → OpenRouter
  Default              → Anthropic

Component-aware / budget-aware requests
----------------------------------------
Callers may tag a request with either headers or a `squeezer_meta` body
field (the body field is stripped before forwarding upstream):

  X-Squeezer-Component: retriever_agent
  X-Squeezer-Run: langgraph-run-8f3c
  X-Squeezer-Budget: 8000

See contextsqueezer.pipeline.component_router for the full mechanism.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
import uuid
from typing import Any

import aiohttp
from aiohttp import web

from contextsqueezer.config import Settings
from contextsqueezer.pipeline.orchestrator import run_pipeline
from contextsqueezer.storage.ccr import CCRManager
from contextsqueezer.storage.sqlite_store import Store, init_db

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Provider routing table
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_upstream(settings: Settings, path: str) -> str:
    if path.startswith("/openai"):
        base = settings.openai_upstream
        tail = path[len("/openai"):]
        return f"{base}{tail}"
    if path.startswith("/openrouter"):
        base = settings.openrouter_upstream
        tail = path[len("/openrouter"):]
        return f"{base}{tail}"
    # Default: Anthropic
    return f"{settings.anthropic_upstream}{path}"


# ──────────────────────────────────────────────────────────────────────────────
# Raw-traffic recorder (opt-in — feeds the offline eval harness)
# ──────────────────────────────────────────────────────────────────────────────

async def _record_raw_payload(settings: Settings, payload: dict, path: str) -> None:
    """
    Append the raw, pre-compression payload to a JSONL file for later offline
    A/B analysis via `squeezer eval run`. Best-effort — never raises.
    """
    try:
        settings.recording_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"path": path, "ts": time.time(), "payload": payload}
        line = json.dumps(record, default=str) + "\n"
        await asyncio.to_thread(_append_line, settings.recording_path, line)
    except Exception as exc:  # pragma: no cover - best-effort, never fatal
        log.debug("Recording failed (non-fatal): %s", exc)


def _append_line(path, line: str) -> None:  # type: ignore[no-untyped-def]
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


# ──────────────────────────────────────────────────────────────────────────────
# CCR tool-call interceptor
# ──────────────────────────────────────────────────────────────────────────────

async def _intercept_tool_call(
    response_body: bytes,
    store: Store,
) -> bytes:
    """
    If the upstream response contains a `squeezer_retrieve` tool call, resolve
    it from the local SQLite store and inject a synthetic tool_result turn.
    """
    try:
        data = json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return response_body

    content = data.get("content", [])
    tool_uses = [
        b for b in content
        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "squeezer_retrieve"
    ]
    if not tool_uses:
        return response_body

    # Resolve each tool call and embed the result in the response
    ccr = CCRManager(store)
    for tool_use in tool_uses:
        result = await ccr.handle_tool_call(tool_use.get("input", {}))
        # Append a synthetic tool_result block
        content.append({
            "type": "tool_result",
            "tool_use_id": tool_use.get("id", ""),
            "content": result,
        })
        log.debug("CCR resolved tool call id=%s", tool_use.get("id"))

    data["content"] = content
    return json.dumps(data).encode()


# ──────────────────────────────────────────────────────────────────────────────
# Main request handler
# ──────────────────────────────────────────────────────────────────────────────

async def _handle(request: web.Request) -> web.Response | web.StreamResponse:
    settings: Settings = request.app["settings"]
    store: Store = request.app["store"]
    session: aiohttp.ClientSession = request.app["session"]

    # ── Read incoming body ────────────────────────────────────────────────────
    try:
        body = await request.read()
        payload = json.loads(body) if body else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return web.Response(status=400, text="Bad Request: invalid JSON")

    upstream_url = _resolve_upstream(settings, request.path_qs)
    t_start = time.perf_counter()

    # ── Opt-in raw traffic recording (for later offline eval) ─────────────────
    if settings.enable_recording:
        asyncio.create_task(_record_raw_payload(settings, copy.deepcopy(payload), request.path_qs))

    # ── Run compression pipeline ──────────────────────────────────────────────
    incoming_headers = dict(request.headers)
    result = await run_pipeline(payload, settings=settings, store=store, headers=incoming_headers)
    log.info(
        "Compressed req_id=%s raw=%d→%d tok (%.0f%%) latency=%.1fms component=%s",
        result.request_id,
        result.raw_tokens,
        result.compressed_tokens,
        result.compression_pct,
        result.proxy_latency_ms,
        result.component_id or "-",
    )

    # ── Forward to upstream ───────────────────────────────────────────────────
    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in (
            "host", "content-length", "transfer-encoding",
            "x-squeezer-component", "x-squeezer-run", "x-squeezer-budget",
        )
    }
    forward_headers["content-type"] = "application/json"

    t_upstream_start = time.perf_counter()
    try:
        async with session.request(
            method=request.method,
            url=upstream_url,
            headers=forward_headers,
            data=json.dumps(result.compressed_payload).encode(),
        ) as upstream_resp:
            upstream_body = await upstream_resp.read()
            upstream_latency_ms = (time.perf_counter() - t_upstream_start) * 1000

            # CCR interception
            if settings.enable_ccr:
                upstream_body = await _intercept_tool_call(upstream_body, store)

            # ── Parse cache hit info from response headers (Anthropic) ─────
            cache_hit = (
                upstream_resp.headers.get("anthropic-cache-creation-input-tokens") is not None
                or upstream_resp.headers.get("x-cache", "") == "HIT"
            )

            # ── Persist metrics ───────────────────────────────────────────────
            asyncio.create_task(
                store.record_metrics(
                    request_id=result.request_id,
                    raw_tokens=result.raw_tokens,
                    compressed_tokens=result.compressed_tokens,
                    proxy_latency_ms=result.proxy_latency_ms,
                    upstream_latency_ms=upstream_latency_ms,
                    algo_breakdown=result.algo_breakdown,
                    cache_hit=cache_hit,
                    ccr_used=result.ccr_used,
                    component_id=result.component_id,
                    run_id=result.run_id,
                    budget_tier=result.budget_tier,
                )
            )

            # ── Return response to local agent ────────────────────────────────
            resp_headers = {
                k: v
                for k, v in upstream_resp.headers.items()
                if k.lower() not in ("transfer-encoding", "content-encoding")
            }
            resp_headers["x-squeezer-tokens-saved"] = str(result.tokens_saved)
            resp_headers["x-squeezer-compression-pct"] = f"{result.compression_pct:.1f}"
            if result.component_id:
                resp_headers["x-squeezer-component"] = result.component_id
            if result.budget_tier >= 0:
                resp_headers["x-squeezer-budget-tier"] = str(result.budget_tier)

            return web.Response(
                status=upstream_resp.status,
                headers=resp_headers,
                body=upstream_body,
            )

    except aiohttp.ClientError as exc:
        log.error("Upstream request failed: %s", exc)
        return web.Response(status=502, text=f"Bad Gateway: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Startup / shutdown
# ──────────────────────────────────────────────────────────────────────────────

async def _startup(app: web.Application) -> None:
    settings: Settings = app["settings"]
    await init_db(settings.db_path)
    app["_store_ctx"] = Store(settings.db_path)
    app["store"] = await app["_store_ctx"].__aenter__()
    connector = aiohttp.TCPConnector(limit=200, ttl_dns_cache=300)
    app["session"] = aiohttp.ClientSession(connector=connector)
    log.info("ContextSqueezer proxy started on %s", settings.proxy_base_url)


async def _shutdown(app: web.Application) -> None:
    await app["session"].close()
    await app["_store_ctx"].__aexit__(None, None, None)
    log.info("ContextSqueezer proxy stopped.")


def build_app(settings: Settings) -> web.Application:
    app = web.Application(client_max_size=32 * 1024 * 1024)  # 32 MB
    app["settings"] = settings
    app.on_startup.append(lambda a: _startup(a))
    app.on_shutdown.append(lambda a: _shutdown(a))
    app.router.add_route("*", "/{path_info:.*}", _handle)
    return app


async def run_proxy(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )
    app = build_app(settings)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, settings.proxy_host, settings.proxy_port)
    await site.start()
    log.info("Proxy listening on %s:%d", settings.proxy_host, settings.proxy_port)
    # Keep running until cancelled
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
