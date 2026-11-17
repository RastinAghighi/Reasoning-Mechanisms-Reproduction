"""Shared command-line entry logic for experiment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

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
    default_config: Path,
    description: str,
    argv: Sequence[str] | None = None,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
) -> tuple[ExperimentConfig, RunContext | None, argparse.Namespace]:
    """Parse args, load config, seed RNGs and (unless --dry-run) create the run dir.

    ``add_arguments`` lets a script register its own flags; the parsed
    namespace is returned as the third element.
    """
    parser = build_parser(default_config, description)
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args(argv)
    config, run = start(args)
    return config, run, args


def start(args: argparse.Namespace) -> tuple[ExperimentConfig, RunContext | None]:
    """Load the config named in parsed args, seed, print a summary, create the run dir."""
    config = load_config(args.config, args.overrides)
    set_seed(config.seed)
    describe(config, args.config)
    if args.dry_run:
        return config, None
    run = init_run(config, args.overrides)
    print(f"run dir    : {run.run_dir}")
    return config, run


def describe(config: ExperimentConfig, source: Path | str) -> None:
    print(f"experiment : {config.name}")
    print(f"config     : {source}  (fingerprint {config.fingerprint()})")
    print(f"seed       : {config.seed}")
    for key, m in config.models.items():
        adapter = f" + adapter {m.adapter_id}" if m.adapter_id else ""
        print(f"model      : {key} = {m.id}@{m.revision or 'UNPINNED'}{adapter}")
    for key, d in config.datasets.items():
        n = d.sample_size if d.sample_size is not None else "all"
        print(f"dataset    : {key} = {d.id}@{d.revision or 'UNPINNED'} splits={list(d.splits)} n={n}")
    if unpinned := config.unpinned():
        print(f"WARNING    : unpinned/missing: {', '.join(unpinned)}")
