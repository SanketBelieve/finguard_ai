"""
database.py — Re-exports ComplianceSQLStore
"""

from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_app_dir = _here / "app"
if _app_dir.is_dir() and str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

from app.db import ComplianceSQLStore

__all__ = ["ComplianceSQLStore"]
