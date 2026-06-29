"""
CLI entry-point for ContextSqueezer.

  squeezer start          – launch proxy (+ optional dashboard)
  squeezer stop           – stop background proxy
  squeezer status         – show running state and quick stats
  squeezer config         – print active configuration
  squeezer config set KEY VALUE
  squeezer flush          – wipe the CCR SQLite store
  squeezer env            – print the env-var snippet to paste into your shell
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from contextsqueezer.config import get_settings

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pid_file() -> Path:
    return get_settings().db_path.parent / "proxy.pid"


def _is_running() -> int | None:
    """Return PID if proxy process is alive, else None."""
    pf = _pid_file()
    if not pf.exists():
        return None
    pid = int(pf.read_text().strip())
    try:
        os.kill(pid, 0)
        return pid
    except (ProcessLookupError, PermissionError):
        pf.unlink(missing_ok=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Root group
# ─────────────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(package_name="contextsqueezer")
def main() -> None:
    """ContextSqueezer – deterministic LLM context compression proxy."""


# ─────────────────────────────────────────────────────────────────────────────
# squeezer start
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--port", default=None, type=int, help="Override proxy port.")
@click.option("--no-dashboard", is_flag=True, default=False, help="Disable analytics dashboard.")
@click.option("--foreground", "-f", is_flag=True, default=False, help="Run in foreground (no daemon).")
def start(port: int | None, no_dashboard: bool, foreground: bool) -> None:
    """Start the ContextSqueezer proxy (and optional dashboard)."""
    settings = get_settings()
    settings.ensure_dirs()

    if port:
        settings.proxy_port = port  # type: ignore[assignment]

    if _is_running():
        console.print(
            f"[yellow]Proxy already running on {settings.proxy_base_url}[/yellow]"
        )
        return

    console.print(
        Panel(
            f"[bold green]Starting ContextSqueezer[/bold green]\n"
            f"  Proxy  → [cyan]{settings.proxy_base_url}[/cyan]\n"
            f"  Dashboard → [cyan]http://127.0.0.1:{settings.dashboard_port}[/cyan]\n"
            f"  DB     → [dim]{settings.db_path}[/dim]",
            title="[bold]ContextSqueezer[/bold]",
            border_style="green",
        )
    )

    if foreground:
        _run_all(settings, no_dashboard)
    else:
        _daemonize(settings, no_dashboard)


def _run_all(settings, no_dashboard: bool) -> None:  # type: ignore[no-untyped-def]
    """Blocking: start proxy (and dashboard) in the current process."""
    from contextsqueezer.proxy.server import run_proxy
    from contextsqueezer.dashboard.server import run_dashboard
    from contextsqueezer.storage.sqlite_store import init_db

    async def _main() -> None:
        await init_db(settings.db_path)
        tasks = [asyncio.create_task(run_proxy(settings))]
        if not no_dashboard and settings.dashboard_enabled:
            tasks.append(asyncio.create_task(run_dashboard(settings)))
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")


def _daemonize(settings, no_dashboard: bool) -> None:  # type: ignore[no-untyped-def]
    """Fork the proxy into the background and write PID file."""
    import subprocess

    cmd = [sys.executable, "-m", "contextsqueezer._worker"]
    if no_dashboard:
        cmd.append("--no-dashboard")
    proc = subprocess.Popen(
        cmd,
        stdout=open(settings.log_path, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _pid_file().write_text(str(proc.pid))
    console.print(f"[green]Proxy started (PID {proc.pid})[/green]")
    console.print(
        f"[dim]Run [bold]squeezer stop[/bold] to terminate.[/dim]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# squeezer stop
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
def stop() -> None:
    """Stop the running proxy daemon."""
    pid = _is_running()
    if pid is None:
        console.print("[yellow]No proxy process found.[/yellow]")
        return
    os.kill(pid, signal.SIGTERM)
    _pid_file().unlink(missing_ok=True)
    console.print(f"[green]Sent SIGTERM to proxy (PID {pid})[/green]")


# ─────────────────────────────────────────────────────────────────────────────
# squeezer status
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
def status() -> None:
    """Show proxy status and quick stats from the SQLite store."""
    settings = get_settings()
    pid = _is_running()

    table = Table(title="ContextSqueezer Status", show_header=False, box=None)
    table.add_column("Key", style="bold")
    table.add_column("Value")

    status_str = (
        f"[green]Running[/green] (PID {pid})" if pid else "[red]Stopped[/red]"
    )
    table.add_row("Status", status_str)
    table.add_row("Proxy", settings.proxy_base_url)
    table.add_row("Dashboard", f"http://127.0.0.1:{settings.dashboard_port}")
    table.add_row("DB", str(settings.db_path))

    if settings.db_path.exists():
        import sqlite3

        with sqlite3.connect(settings.db_path) as conn:
            try:
                rows = conn.execute(
                    "SELECT COUNT(*), SUM(raw_tokens), SUM(compressed_tokens) FROM metrics"
                ).fetchone()
                if rows[0]:
                    saved = (rows[1] or 0) - (rows[2] or 0)
                    pct = (saved / rows[1] * 100) if rows[1] else 0
                    table.add_row("Requests processed", str(rows[0]))
                    table.add_row("Tokens saved", f"{saved:,} ({pct:.1f}%)")
                ccr_rows = conn.execute("SELECT COUNT(*) FROM ccr_store").fetchone()
                table.add_row("CCR entries", str(ccr_rows[0]))
            except sqlite3.OperationalError:
                table.add_row("DB", "[dim]Not yet initialised[/dim]")

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# squeezer env
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
def env() -> None:
    """Print environment variable snippet to route agent traffic through proxy."""
    s = get_settings()
    snippet = (
        f"# ── Paste into your shell or .env ────────────────────────\n"
        f"export ANTHROPIC_BASE_URL='{s.proxy_base_url}'\n"
        f"export OPENAI_BASE_URL='{s.proxy_base_url}/openai'\n"
        f"export OPENROUTER_BASE_URL='{s.proxy_base_url}/openrouter'\n"
        f"# ──────────────────────────────────────────────────────────"
    )
    console.print(Panel(snippet, title="[bold]Environment Setup[/bold]", border_style="cyan"))


# ─────────────────────────────────────────────────────────────────────────────
# squeezer config
# ─────────────────────────────────────────────────────────────────────────────

@main.group()
def config() -> None:
    """View or modify ContextSqueezer configuration."""


@config.command("show")
def config_show() -> None:
    """Print the active configuration."""
    s = get_settings()
    table = Table(title="Active Configuration", show_header=True)
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value")

    for field_name, value in s.model_dump().items():
        table.add_row(field_name, str(value))

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# squeezer flush
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
@click.confirmation_option(prompt="This will wipe ALL CCR entries and metrics. Continue?")
def flush() -> None:
    """Wipe the CCR store and metrics tables."""
    import sqlite3

    settings = get_settings()
    if not settings.db_path.exists():
        console.print("[yellow]Nothing to flush.[/yellow]")
        return

    with sqlite3.connect(settings.db_path) as conn:
        conn.execute("DELETE FROM ccr_store")
        conn.execute("DELETE FROM metrics")
        conn.commit()

    console.print("[green]Store flushed.[/green]")


# ─────────────────────────────────────────────────────────────────────────────
# squeezer eval
# ─────────────────────────────────────────────────────────────────────────────

@main.group()
def eval() -> None:
    """Test compression against real or recorded traffic."""


@eval.command("run")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--live", is_flag=True, default=False,
              help="Also replay against the real Anthropic API (needs ANTHROPIC_API_KEY).")
@click.option("--limit", type=int, default=None, help="Only process the first N cases.")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Write the full JSON report to this path.")
def eval_run(path: Path, live: bool, limit: int | None, out: Path | None) -> None:
    """
    Run the compression pipeline over a JSONL file of recorded/sample requests
    and report token savings (and, with --live, answer-similarity) per case.

    PATH can be:
      • a file recorded by the proxy (SQUEEZER_ENABLE_RECORDING=true), or
      • a hand-built JSONL of Anthropic-messages-format payloads — see
        contextsqueezer/eval/fixtures/sample_coding_session.jsonl
    """
    import asyncio  # noqa: F811 - already module-level, kept local for clarity
    from contextsqueezer.eval.harness import run_eval_from_file

    settings = get_settings()

    if live and not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[red]--live requires ANTHROPIC_API_KEY to be set in the environment.[/red]"
        )
        return

    console.print(f"[dim]Running eval over {path} (live={live})...[/dim]")
    report = asyncio.run(
        run_eval_from_file(path, settings=settings, live=live, limit=limit)
    )

    table = Table(title="ContextSqueezer Eval Report", show_header=True)
    table.add_column("Case", justify="right")
    table.add_column("Raw tok", justify="right")
    table.add_column("Compressed tok", justify="right")
    table.add_column("Saved %", justify="right")
    table.add_column("Similarity", justify="right")
    table.add_column("Note")

    for c in report.cases:
        note = c.error or ""
        sim = f"{c.live_similarity:.2f}" if c.live_similarity is not None else "—"
        sim_style = ""
        if c.live_similarity is not None and c.live_similarity < 0.5:
            sim_style = "[red]"
            note = note or "answer drift — inspect manually"
        table.add_row(
            str(c.index),
            str(c.raw_tokens),
            str(c.compressed_tokens),
            f"{c.compression_pct:.1f}%",
            f"{sim_style}{sim}",
            note,
        )

    console.print(table)
    console.print(
        Panel(
            f"Total: [bold]{report.total_raw_tokens:,}[/bold] → "
            f"[bold]{report.total_compressed_tokens:,}[/bold] tokens "
            f"([green]{report.overall_compression_pct:.1f}% saved[/green])\n"
            f"Avg proxy latency: {report.avg_proxy_latency_ms:.2f}ms\n"
            f"Per-algorithm: {report.algo_totals}",
            title="Summary",
            border_style="cyan",
        )
    )

    if report.low_similarity_cases:
        console.print(
            f"[red]⚠ {len(report.low_similarity_cases)} case(s) showed low answer "
            f"similarity (<0.5) — inspect these before trusting compression on "
            f"similar traffic.[/red]"
        )

    if out:
        out.write_text(json.dumps(report.to_dict(), indent=2))
        console.print(f"[dim]Full report written to {out}[/dim]")
