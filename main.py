"""Modular RAG MCP Server - main entry point.

stdout is reserved for MCP protocol messages, so every diagnostic goes to stderr.
"""

import sys
from pathlib import Path

# MCP clients launch the server as a subprocess with an arbitrary working
# directory, so make the source root importable without relying on an install.
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

__version__ = "0.1.0"


def main() -> int:
    """Start the MCP Server. Returns the process exit code."""
    print(f"Modular RAG MCP Server {__version__} - skeleton ready.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
