"""
FinGuard AI Service Package
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "app":
        from app.main import app as _app

        globals()["app"] = _app
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app"]
