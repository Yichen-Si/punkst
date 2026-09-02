#!/usr/bin/env python3
"""Checkout entry point for ``punkst-multires``."""

from pathlib import Path
import sys

PYTHON_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PYTHON_ROOT))

from punkst_multires.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
