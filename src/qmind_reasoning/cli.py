"""Shared command-line entry logic for experiment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from qmind_reasoning.config import ExperimentConfig, load_config
from qmind_reasoning.runs import RunContext, init_run
from qmind_reasoning.seeding import set_seed


def build_parser(default_config: Path, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=default_config, help="YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set datasets.processbench.sample_size=20 (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and print the config without creating a run directory",
    )
    return parser


def prepare(
    default_config: Path, description: str, argv: Sequence[str] | None = None
) -> tuple[ExperimentConfig, RunContext | None]:
    """Parse args, load config, seed RNGs and (unless --dry-run) create the run dir."""
    args = build_parser(default_config, description).parse_args(argv)
    config = load_config(args.config, args.overrides)
    set_seed(config.seed)

    print(f"experiment : {config.name}")
    print(f"config     : {args.config}  (fingerprint {config.fingerprint()})")
    print(f"seed       : {config.seed}")
    for key, m in config.models.items():
        adapter = f" + adapter {m.adapter_id}" if m.adapter_id else ""
        print(f"model      : {key} = {m.id}@{m.revision or 'UNPINNED'}{adapter}")
    for key, d in config.datasets.items():
        n = d.sample_size if d.sample_size is not None else "all"
        print(f"dataset    : {key} = {d.id}@{d.revision or 'UNPINNED'} splits={list(d.splits)} n={n}")
    if unpinned := config.unpinned():
        print(f"WARNING    : unpinned/missing: {', '.join(unpinned)}")

    if args.dry_run:
        return config, None
    run = init_run(config, args.overrides)
    print(f"run dir    : {run.run_dir}")
    return config, run
