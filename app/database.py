"""
database.py — Re-exports ComplianceSQLStore from db.py
======================================================
Provides backward and cross-module compatibility for imports
expecting `database.ComplianceSQLStore`.
"""

from __future__ import annotations

from db import ComplianceSQLStore

__all__ = ["ComplianceSQLStore"]
