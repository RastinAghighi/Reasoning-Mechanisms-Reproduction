"""Canonical repository locations."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` (default: this file) to the dir holding pyproject.toml."""
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"no pyproject.toml found above {here}")


PROJECT_ROOT = find_project_root()
CONFIGS_DIR = PROJECT_ROOT / "configs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"


def resolve_output_root(output_root: str | Path) -> Path:
    """Relative output roots are interpreted relative to the project root."""
    path = Path(output_root)
    return path if path.is_absolute() else PROJECT_ROOT / path
