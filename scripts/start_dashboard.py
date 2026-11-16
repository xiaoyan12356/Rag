#!/usr/bin/env python
"""Dashboard script for the Modular RAG MCP Server.

This script starts the dashboard, which is the visual side of the server:
the configuration, the ingested documents, the ingestion runs and the
questions that were asked.

Usage:
    # Start the dashboard on the default port
    python scripts/start_dashboard.py

    # Start it on another port
    python scripts/start_dashboard.py --port 8502

    # Bind it to another address
    python scripts/start_dashboard.py --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

if sys.platform == "win32":  # pragma: no cover - depends on the terminal
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

APP_PATH = _REPO_ROOT / "src" / "observability" / "dashboard" / "app.py"
EXIT_FAILED = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the arguments of the command line."""
    parser = argparse.ArgumentParser(
        description="Start the dashboard of the Modular RAG MCP Server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8501,
        help="The port to serve the dashboard on (default: 8501)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="The address to bind to (default: localhost)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Start the dashboard and say how it went."""
    args = parse_args(argv)

    if not APP_PATH.exists():
        print(f"[FAIL] The dashboard application is not there: {APP_PATH}")
        return EXIT_FAILED

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
        "--server.headless",
        "true",
    ]
    print(f"[INFO] Starting the dashboard on http://{args.host}:{args.port}")
    print("[INFO] Press Ctrl+C to stop it.")

    try:
        completed = subprocess.run(command, cwd=str(_REPO_ROOT))
    except KeyboardInterrupt:
        print("\n[INFO] The dashboard was stopped.")
        return 0
    except OSError as exc:
        print(f"[FAIL] The dashboard could not be started: {exc}")
        return EXIT_FAILED
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
