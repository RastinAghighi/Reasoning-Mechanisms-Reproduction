"""Run directories and run metadata.

Every execution of an experiment gets its own directory::

    results/<experiment>/<YYYYmmdd-HHMMSS>_s<seed>_<fingerprint>/
        config.yaml      # fully resolved config (after inheritance + overrides)
        metadata.json    # provenance: git state, versions, command, status
        artifacts/       # predictions, scores, tables, plots
        logs/

so that any number can be traced back to the exact config and code that made it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable

import yaml

from qmind_reasoning.config import ExperimentConfig
from qmind_reasoning.paths import PROJECT_ROOT, resolve_output_root

TRACKED_PACKAGES = ("pyyaml", "numpy", "torch", "transformers", "datasets", "accelerate", "peft")
RUN_SUBDIRS = ("artifacts", "logs")


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    config: ExperimentConfig

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "metadata.json"

    def read_metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def update_metadata(self, **fields: Any) -> None:
        """Merge fields into metadata.json (e.g. status, metrics summary)."""
        data = self.read_metadata()
        data.update(fields)
        _write_json(self.metadata_path, data)

    def finish(self, status: str = "completed", **fields: Any) -> None:
        self.update_metadata(status=status, finished_utc=_utc_now(), **fields)


def init_run(
    config: ExperimentConfig,
    overrides: Iterable[str] = (),
    output_root: str | Path | None = None,
) -> RunContext:
    """Create a fresh run directory and write the resolved config + metadata."""
    root = resolve_output_root(output_root or config.output_root) / config.name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base_id = f"{stamp}_s{config.seed}_{config.fingerprint()}"

    run_id, suffix = base_id, 1
    while (root / run_id).exists():
        suffix += 1
        run_id = f"{base_id}-{suffix}"

    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    for sub in RUN_SUBDIRS:
        (run_dir / sub).mkdir()

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.to_dict(), fh, sort_keys=False)

    _write_json(run_dir / "metadata.json", collect_metadata(config, run_id, list(overrides)))
    return RunContext(run_id=run_id, run_dir=run_dir, config=config)


def collect_metadata(config: ExperimentConfig, run_id: str, overrides: list[str]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment": config.name,
        "status": "initialized",
        "created_utc": _utc_now(),
        "seed": config.seed,
        "config_fingerprint": config.fingerprint(),
        "overrides": overrides,
        "unpinned_revisions": config.unpinned(),
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]] if sys.argv else [],
        "git": git_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
    }


def git_state(repo: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Commit hash and dirty flag; degrades gracefully outside a git repo."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"available": False}
    status = run("status", "--porcelain")
    return {"available": True, "commit": commit, "dirty": bool(status)}


def package_versions(names: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed versions via package metadata (does not import the packages)."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
