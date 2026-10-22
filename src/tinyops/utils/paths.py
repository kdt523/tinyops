"""Repo-relative paths."""

from __future__ import annotations

from pathlib import Path


def _find_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate repo root (no pyproject.toml in any parent folder)")


ROOT = _find_root()
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RUNS = ROOT / "runs"
CONFIGS = ROOT / "configs"
DOCS = ROOT / "docs"
