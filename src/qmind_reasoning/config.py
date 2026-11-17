"""YAML-based experiment configuration.

A config file may inherit from another via a top-level ``extends`` key (path
relative to the file). Parent and child are deep-merged, child wins. Values can
then be overridden from the command line with dotted ``key=value`` strings,
e.g. ``seed=1`` or ``datasets.processbench.sample_size=50``.

The resolved config is validated into frozen dataclasses so that typos and
missing fields fail loudly before any compute is spent.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class ConfigError(ValueError):
    """Raised for malformed or invalid configuration."""


@dataclass(frozen=True)
class ModelSpec:
    """A Hugging Face model, optionally with a PEFT adapter on top."""

    id: str | None
    revision: str | None = None  # commit hash / tag; pin before final runs
    dtype: str = "auto"
    adapter_id: str | None = None
    adapter_revision: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    """A Hugging Face dataset and the subset of it a run uses."""

    id: str | None
    revision: str | None = None
    config_name: str | None = None
    splits: tuple[str, ...] = ()
    sample_size: int | None = None  # None = use every example


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str = ""
    seed: int = 0
    output_root: str = "results"
    models: dict[str, ModelSpec] = field(default_factory=dict)
    datasets: dict[str, DatasetSpec] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)  # experiment-specific knobs
    tags: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation (tuples become lists)."""
        return json.loads(json.dumps(asdict(self)))

    def fingerprint(self) -> str:
        """Short, stable hash of the resolved config, for run identification."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def unpinned(self) -> list[str]:
        """Names of models/datasets/adapters that lack a pinned revision or ID."""
        missing = []
        for key, m in self.models.items():
            if m.id is None:
                missing.append(f"models.{key}.id")
            elif m.revision is None:
                missing.append(f"models.{key}.revision")
            if m.adapter_id is not None and m.adapter_revision is None:
                missing.append(f"models.{key}.adapter_revision")
        for key, d in self.datasets.items():
            if d.id is None:
                missing.append(f"datasets.{key}.id")
            elif d.revision is None:
                missing.append(f"datasets.{key}.revision")
        return missing


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_config(path: str | Path, overrides: Iterable[str] = ()) -> ExperimentConfig:
    """Load, merge, override and validate a config file."""
    raw = _load_with_inheritance(Path(path).resolve(), seen=())
    for item in overrides:
        _apply_override(raw, item)
    return from_dict(raw)


def from_dict(raw: Mapping[str, Any]) -> ExperimentConfig:
    data = dict(raw)
    _check_keys(data, ExperimentConfig, "config")
    if not data.get("name"):
        raise ConfigError("config: 'name' is required")

    models = {}
    for key, spec in (data.pop("models", None) or {}).items():
        _check_keys(spec, ModelSpec, f"models.{key}")
        models[key] = ModelSpec(**spec)

    datasets = {}
    for key, spec in (data.pop("datasets", None) or {}).items():
        _check_keys(spec, DatasetSpec, f"datasets.{key}")
        spec = dict(spec)
        spec["splits"] = tuple(spec.get("splits") or ())
        size = spec.get("sample_size")
        if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size <= 0):
            raise ConfigError(f"datasets.{key}.sample_size must be a positive int or null, got {size!r}")
        datasets[key] = DatasetSpec(**spec)

    seed = data.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(f"config: 'seed' must be an int, got {seed!r}")

    data["tags"] = tuple(data.get("tags") or ())
    data["params"] = dict(data.get("params") or {})
    return ExperimentConfig(models=models, datasets=datasets, **data)


def _load_with_inheritance(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    if path in seen:
        raise ConfigError(f"circular 'extends' chain: {' -> '.join(map(str, seen + (path,)))}")
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    parent = data.pop("extends", None)
    if parent is None:
        return data
    base = _load_with_inheritance((path.parent / parent).resolve(), seen + (path,))
    return _deep_merge(base, data)


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _apply_override(raw: dict[str, Any], item: str) -> None:
    if "=" not in item:
        raise ConfigError(f"override must look like key=value, got {item!r}")
    dotted, value_text = item.split("=", 1)
    keys = dotted.strip().split(".")
    node = raw
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override {dotted!r}: {key!r} is not a mapping")
    node[keys[-1]] = yaml.safe_load(value_text)  # "50" -> 50, "null" -> None, "[a,b]" -> list


def _check_keys(data: Any, cls: type, where: str) -> None:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
