#!/usr/bin/env python3
"""Точка входа калькуляторов: python3 scripts/calc.py <команда> ..."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from infra_calc.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
