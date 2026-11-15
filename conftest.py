"""Pytest bootstrap: make the src/ packages importable without an editable install."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Shared contract suites live outside the test tree and are imported by name.
CONTRACT_DIR = ROOT_DIR / "tests" / "contract"
if str(CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTRACT_DIR))
