"""Modular RAG MCP Server - main entry point.

stdout is reserved for MCP protocol messages, so every diagnostic goes to stderr.
"""

import argparse
import sys
from pathlib import Path

# MCP clients launch the server as a subprocess with an arbitrary working
# directory, so make the source root importable without relying on an install.
PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mcp_server.server import main as serve  # noqa: E402

__version__ = "0.1.0"


def main() -> int:
    """Start the MCP Server. Returns the process exit code."""
    return serve(_server_arguments())


def _server_arguments() -> list[str]:
    """Return the arguments the server understands.

    A client may start the server with arguments of its own, so the ones
    that are not ours are left out instead of being refused.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("--mode", choices=("sdk", "plain"))
    known, _ = parser.parse_known_args()

    arguments: list[str] = []
    if known.config:
        arguments += ["--config", known.config]
    if known.mode:
        arguments += ["--mode", known.mode]
    return arguments


if __name__ == "__main__":
    sys.exit(main())
