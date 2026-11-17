from pathlib import Path

import pytest


@pytest.fixture
def write_yaml(tmp_path: Path):
    """Write a YAML string to tmp_path/<name> and return its path."""

    def _write(name: str, text: str) -> Path:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    return _write
