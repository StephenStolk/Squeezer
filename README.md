# ContextSqueezer

A local, deterministic context-compression proxy for LLM developer agents (Claude Code, Cursor, Aider, LangChain/LangGraph pipelines, multi-agent swarms). It sits between your agent(s) and the model provider, shrinks the outgoing payload using rule-based (zero-ML) techniques, and forwards the compressed request — with a local SQLite fallback so the model can still pull back anything it actually needs.

No remote calls, no telemetry, no inference cost added by the proxy itself. Everything happens on `localhost`.

```
Your Agent(s)  →  ContextSqueezer (localhost:8787)  →  Anthropic / OpenAI / OpenRouter
                          │
                          ├── Squeezer-meta extraction (component_id / run_id / budget)
                          ├── PII scrubber (regex + entropy, local only)
                          ├── LSH cross-turn deduplicator (SimHash + Rabin fingerprint)
                          ├── Temporal context decay (pin-aware)
                          ├── Per-message engines, budget-gated:
                          │     ├── Cross-component dedup (CCR-backed, retrievable)
                          │     ├── File-version delta encoding (diff vs. last read)
                          │     ├── AST compactor (tree-sitter, strips non-focal bodies)
                          │     ├── JSON smart crusher (depth clamp + array truncation)
                          │     ├── Shell-output minifier (drops passing tests, etc.)
                          │     └── Linguistic minifier (strips filler, compacts prose)
                          ├── Cache-aligner v2 (stable, append-only prefix — provider-cache safe)
                          └── CCR (Content-Compressed Retrieval) — SQLite + squeezer_retrieve tool
```

## What's actually different here vs. a generic compressor

This isn't trying to out-compress anyone on raw ratio. It targets three gaps that are easy to miss when you're building one tool for one linear conversation:

1. **Multi-component awareness.** Real systems aren't one chat thread — they're a LangGraph graph, a swarm of agents, several services that each call the same model. Independent components routinely send *overlapping* context (the same retrieved chunk, the same tool result) without knowing another component already paid for it seconds earlier. Tag requests with a shared `run_id` and per-call `component_id`, and the proxy catches that redundancy across components, not just across turns. See [Component-aware proxying](#component-aware-proxying-the-multi-agent-case).
2. **Provider-cache-safe by construction, not by luck.** v1 of the cache aligner (kept as the cautionary tale it is — see commit history) re-sorted messages by a freshly-computed "stability score" every request. That can silently break Anthropic/OpenAI's exact-prefix cache matching the moment a message shifts buckets between calls. v2 does no content-based reordering at all — it relies on the conversation's natural append-only growth and places the cache breakpoint at a fixed relative offset, which is what actually produces a stable, growing, cache-matchable prefix turn over turn.
3. **History-aware delta encoding for files, not just turn-to-turn similarity.** LSH dedup only catches *identical or near-identical* repeats inside one conversation. Agents re-reading the same file after a small edit is the single most common redundancy pattern in real coding sessions, and it isn't "near-duplicate text," it's "95% the same file with a tracked version history." `file_version_tracker.py` keeps a per-file version chain and sends a unified diff (or a zero-content pointer if nothing changed) instead of the whole file again.

## Install

```bash
pip install -e ".[dev]"          # from a clone, editable + dev deps
```

> **Tree-sitter note:** pinned to `tree-sitter<0.22` because `tree-sitter-languages` hasn't been updated for the 0.22+ `Language` API. See [Known Limitations](#known-limitations).

## Quick start

```bash
squeezer start                        # background daemon + dashboard
eval "$(squeezer env)"                # exports ANTHROPIC_BASE_URL etc. for this shell
# run your agent as normal — anything that respects ANTHROPIC_BASE_URL
open http://127.0.0.1:8788            # watch savings live
squeezer stop
```

`squeezer start --foreground` runs without daemonizing, logs to stdout.

## CLI reference

| Command | What it does |
|---|---|
| `squeezer start [--port N] [--no-dashboard] [-f]` | Start proxy (+ dashboard) |
| `squeezer stop` | Stop the background daemon |
| `squeezer status` | Show running state + quick token-savings stats |
| `squeezer env` | Print env-var snippet for routing agent traffic |
| `squeezer config show` | Print active configuration |
| `squeezer flush` | Wipe CCR store + metrics (with confirmation) |
| `squeezer eval run PATH [--live] [--limit N] [--out FILE]` | Test against real/recorded traffic — see below |

All settings are env vars prefixed `SQUEEZER_` — see `.env.example`.

## Component-aware proxying (the multi-agent case)

If your "big project" is actually several components — a retriever, a planner, a critic, whatever — that each independently call the LLM, tag each call with a shared `run_id` and a per-component `component_id`. Either headers or a body field works; headers take precedence and the body field is stripped before forwarding upstream:

```bash
curl $ANTHROPIC_BASE_URL/v1/messages \
  -H "X-Squeezer-Component: retriever_agent" \
  -H "X-Squeezer-Run: langgraph-run-8f3c" \
  -d '{"model": "claude-sonnet-4-6", "messages": [...]}'
```

```python
# or via the body, if your client makes header injection awkward
payload = {
    "model": "claude-sonnet-4-6",
    "messages": [...],
    "squeezer_meta": {"component_id": "planner_agent", "run_id": "langgraph-run-8f3c"},
}
```

When a second component sends content (≥200 chars) that another component already sent within the *same* `run_id`, it's stored via the same CCR mechanism used for size-based offloading and replaced with a real `[CCR:...]` pointer — `squeezer_retrieve`-able, not a dead end — instead of being resent and recompressed from scratch. This is local, in-process, and scoped to the run: different `run_id`s never share state, and a component never dedupes against its own earlier sends (that's what the LSH turn-deduplicator is for, within a single growing conversation).

This is genuinely useful, and genuinely a real engineering surface — it's also new and not yet been battle-tested against a production multi-agent workload the way the rest of this is. Validate it against your own LangGraph/CrewAI/etc. setup before trusting it broadly.

## Budget-aware adaptive compression

Most rule-based compressors apply one fixed aggressiveness to everything. Declare a target instead, and the pipeline only escalates as far as it needs to:

```python
payload["squeezer_meta"] = {"budget_tokens": 8000}
# or: -H "X-Squeezer-Budget: 8000"
```

If the request is already under budget, the pipeline runs light (PII scrub + exact dedup only) to preserve maximum fidelity. If it's way over, it escalates through AST stripping, shell minification, temporal decay, and a much lower CCR offload threshold — whatever it takes, capped by what your global `SQUEEZER_ENABLE_*` settings allow. A budget can only ever make things *more* conservative than your global settings already permit; it can't switch on something you've globally disabled.

## Pinning content against temporal decay

Old turns get condensed to a keyword digest by the temporal-decay stage. If something genuinely needs to survive regardless of age (a deployment target, a user preference stated once early on), prefix that message's content with `[PIN]`:

```python
{"role": "user", "content": "[PIN] Always deploy to the staging cluster, never prod, until I say otherwise."}
```

The marker is stripped before the request goes upstream — the model never sees it, it just sees the content kept verbatim no matter how old the turn gets.

## Testing this against real stuff

This is the part that actually matters — a compressor that "saves tokens" on toy examples and has never been pointed at real traffic is an unverified claim, not a result. Here's how to verify it on your own workload:

### 1. Get real data

**Option A — record live traffic from your actual agent(s):**

```bash
SQUEEZER_ENABLE_RECORDING=true squeezer start
eval "$(squeezer env)"
# run your real agent — Claude Code, your LangGraph pipeline, whatever —
# through a normal session, doing real work
```

Every raw, pre-compression request gets appended as one JSON line to `~/.config/contextsqueezer/recordings/raw_requests.jsonl`. This is the actual shape of your actual traffic, not a guess.

**Option B — hand-build a JSONL file** of Anthropic-messages-format payloads. See `contextsqueezer/eval/fixtures/sample_coding_session.jsonl` for the expected shape (one JSON object per line, each with `model`/`messages`/optionally `system`).

### 2. Run the offline report (free, no API calls)

```bash
squeezer eval run ~/.config/contextsqueezer/recordings/raw_requests.jsonl
```

This runs the real pipeline over your real recorded requests and reports raw vs. compressed tokens, per-algorithm breakdown, and proxy latency overhead — per case and in aggregate. Try it right now against the bundled synthetic sample:

```bash
squeezer eval run contextsqueezer/eval/fixtures/sample_coding_session.jsonl
```

### 3. Check for answer drift (costs real API calls)

```bash
export ANTHROPIC_API_KEY=sk-...
squeezer eval run recordings/raw_requests.jsonl --live --limit 10
```

`--live` replays a sample of requests against the *real* Anthropic API both with and without compression, and reports a lexical-similarity score (`difflib.SequenceMatcher`, deterministic, no embedding model, no judge call — deliberately simple) between the two answers. A case flagged with similarity below 0.5 means "go read both transcripts yourself" — this check is a tripwire for gross drift, not a rigorous equivalence proof. Don't take a high score as a correctness guarantee either; it just means the two answers were lexically similar, not that either one was *right*.

### 4. What "good" looks like

Run this on a real session, not the synthetic sample, before you put a number on a resume or a README badge. A defensible claim looks like: *"On N real requests recorded from my actual [project] agent, ContextSqueezer reduced average input tokens by X%, with zero flagged answer-drift cases in a --live sample of M."* That's a sentence backed by a command anyone can rerun. "Saves up to 95% of tokens" with no recorded-traffic numbers behind it is the kind of claim this README is explicitly trying not to make.

## Dashboard

FastAPI + vanilla Chart.js, single HTML file, no build step, served at `:8788`:

- Token timeline (raw vs. compressed)
- Per-algorithm savings breakdown (doughnut + table)
- **Per-component breakdown** — which agent/service in your multi-component system is actually the token hog (only appears once you start tagging requests with `component_id`)
- Cache-hit / CCR-fetch counters
- Local-only PII interception audit log

## Project layout

```
contextsqueezer/
├── cli.py                          squeezer start/stop/status/env/config/flush/eval
├── config.py                       all SQUEEZER_* settings
├── proxy/server.py                 aiohttp reverse proxy, recording, tool-call interception
├── pipeline/
│   ├── orchestrator.py             9-step pipeline, MIXED-content block splitting
│   ├── classifier.py               CODE/JSON/SHELL/CONVERSATION/MIXED + block splitter
│   ├── cache_aligner.py            v2 — stable-prefix-safe, no content reordering
│   ├── component_router.py         cross-component dedup (the multi-agent feature)
│   └── budget_governor.py          declarative compression-intensity tiers
├── compressors/
│   ├── ast_compactor.py            tree-sitter skeleton + regex Python fallback
│   ├── call_graph_pruner.py        stdlib-ast BFS reachability (Python-only — see limitations)
│   ├── file_version_tracker.py     diff-based delta encoding for repeated file reads
│   ├── json_crusher.py             depth clamp + Kneedle elbow + array truncation
│   ├── lsh_deduplicator.py         64-bit SimHash + Rabin fingerprint
│   ├── shell_sandbox.py           passing-test strip, Levenshtein grouping
│   ├── linguistic_minifier.py     filler/sycophancy strip, phrase compaction
│   └── temporal_decay.py          recent/partial/keyword-digest aging zones, pin-aware
├── storage/
│   ├── sqlite_store.py             WAL-mode SQLite — metrics, CCR, file versions, xcomp ledger
│   └── ccr.py                      offload/pointer/resolve + squeezer_retrieve tool
├── security/pii_scrubber.py        regex bank + Shannon-entropy sweep
├── dashboard/server.py             FastAPI + inline Chart.js dashboard, :8788
└── eval/
    ├── harness.py                  offline + --live A/B testing against real traffic
    └── fixtures/sample_coding_session.jsonl
```

## Testing (unit/integration suite)

```bash
pip install -e ".[dev]"
pytest -v                                  # 72 tests
pytest --cov=contextsqueezer --cov-report=term-missing
```

## Known Limitations

Read this before you trust it with anything that matters, or claim a number from it without checking.

1. **Call-graph pruning is Python-only.** `call_graph_pruner.py` uses Python's stdlib `ast` — no JS/TS/Go/Rust call-graph reachability yet, even though the AST compactor handles those languages for in-file body stripping.
2. **`tree-sitter-languages` is effectively unmaintained** and incompatible with `tree-sitter>=0.22`. Pinned for now; migrate to `tree-sitter-language-pack` when you have a free afternoon.
3. **Stripped/diffed/pointed-to content is lossy by construction.** AST bodies, JSON depth, file diffs, cross-component pointers, temporal-decay digests — none of it is force-fed back to the model. If the model doesn't realize it needs the missing piece, it won't ask, and it will generate fluent, confident, wrong output. There's no mechanism that *forces* a `squeezer_retrieve` call; it's opt-in from the model's side.
4. **Cross-component dedup is new and unvalidated at scale.** The mechanism is straightforward (a run-scoped content-hash ledger) but it hasn't been run against a real multi-agent production workload yet. Validate it against your own setup — check that components which *should* see fresh content (not a stale pointer to something another agent saw under different framing) aren't getting incorrectly deduped.
5. **The MIXED-content classifier is heuristic, not exhaustive.** `split_into_blocks` now sniffs fenced-block content instead of assuming every ``` fence is code, but the underlying signal is still regex-based pattern matching, not a real parser. Edge cases will misclassify.
6. **CCR retrieval still adds round-trip risk.** Resolving a `squeezer_retrieve` call (whether for a size-based CCR offload or a cross-component dedup hit — both now route through the same retrievable storage) costs an extra request/response cycle. For latency-sensitive interactive use against fast models, that round trip can outweigh the token-cost savings.
7. **Regex/entropy PII scrubbing has real false-negative and false-positive rates.** Same caveat as always — it will miss some real secrets and occasionally redact legitimate high-entropy strings.
8. **JSON crushing and file-diff fallback are both genuinely lossy.** Depth-clamped JSON fields and "rewrite, don't diff" fallbacks both discard information rather than summarizing it.
9. **No streaming support.** Full request/response bodies are buffered; SSE/streaming upstream responses aren't passed through incrementally.
10. **The `--live` eval mode is a tripwire, not a proof.** Lexical similarity via `difflib` catches gross drift; it says nothing about which answer (if either) was actually correct.

If any of these matter for your use case, fix them before you rely on this in production — and if you're citing this on a resume, cite it with eval-harness numbers from your *own* recorded traffic (see [Testing this against real stuff](#testing-this-against-real-stuff)), not a number from this README.
