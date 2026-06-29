"""
Internal worker module.
Launched by `squeezer start` (without --foreground) as a background subprocess.
"""

from __future__ import annotations

import argparse
import asyncio

from contextsqueezer.config import get_settings
from contextsqueezer.proxy.server import run_proxy
from contextsqueezer.dashboard.server import run_dashboard
from contextsqueezer.storage.sqlite_store import init_db


async def _main(no_dashboard: bool) -> None:
    settings = get_settings()
    await init_db(settings.db_path)
    tasks = [asyncio.create_task(run_proxy(settings))]
    if not no_dashboard and settings.dashboard_enabled:
        tasks.append(asyncio.create_task(run_dashboard(settings)))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dashboard", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main(args.no_dashboard))
