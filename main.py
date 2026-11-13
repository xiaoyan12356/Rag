"""Modular RAG MCP Server - main entry point.

stdout is reserved for MCP protocol messages, so every diagnostic goes to stderr.
"""

import os
import sys
from pathlib import Path

# MCP clients launch the server as a subprocess with an arbitrary working
# directory, so make the source root importable without relying on an install.
PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from core.settings import SettingsError, load_settings  # noqa: E402
from observability.logger import get_logger  # noqa: E402

__version__ = "0.1.0"

CONFIG_PATH_ENV_VAR = "SMART_KNOWLEDGE_HUB_CONFIG"


def settings_path() -> Path:
    """Locate the configuration file: env override first, then the checkout."""
    override = os.environ.get(CONFIG_PATH_ENV_VAR)
    if override:
        return Path(override)
    return PROJECT_ROOT / "config" / "settings.yaml"


def main() -> int:
    """Start the MCP Server. Returns the process exit code."""
    logger = get_logger()

    try:
        settings = load_settings(settings_path())
    except SettingsError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    logger.info("Settings loaded from %s", settings_path())
    logger.info(
        "llm=%s/%s embedding=%s/%s vector_store=%s",
        settings.llm.provider,
        settings.llm.model,
        settings.embedding.provider,
        settings.embedding.model,
        settings.vector_store.provider,
    )
    logger.info("MCP Server will be implemented in Phase E.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
