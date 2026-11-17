"""Experiment 1, step 1: prepare and validate ProcessBench data.

Loads each configured split at the pinned revision, validates the schema and
label semantics, draws a deterministic sample, and writes to the run directory:

    artifacts/processbench_<split>_manifest.json   revision, sha256, seed, sample IDs,
                                                   validation results, statistics
    artifacts/processbench_<split>_sample.jsonl    selected records, verbatim
    artifacts/processbench_<split>_report.md       human-readable summary

Usage:
    python experiments/01_error_localization/prepare_data.py --config configs/01_error_localization_smoke.yaml
    python experiments/01_error_localization/prepare_data.py --config configs/01_error_localization_dev.yaml
    python experiments/01_error_localization/prepare_data.py            # full split

Exit code is 0 only if every split validates and matches the paper's counts.
No model weights are loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qmind_reasoning.cli import prepare
from qmind_reasoning.data import processbench as pb
from qmind_reasoning.paths import CONFIGS_DIR

DEFAULT_CONFIG = CONFIGS_DIR / "01_error_localization.yaml"
DATASET_KEY = "processbench"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="use a local split file instead of downloading (single split only)",
    )


def main(argv: list[str] | None = None) -> int:
    config, run, args = prepare(DEFAULT_CONFIG, __doc__.splitlines()[0], argv, add_arguments)
    spec = config.datasets[DATASET_KEY]
    if args.data_file is not None and len(spec.splits) != 1:
        raise SystemExit("--data-file requires exactly one configured split")
    if run is None:
        return 0

    stratify = bool(config.params.get("stratify_sampling", True))
    summaries, all_passed = {}, True
    for split in spec.splits:
        manifest, sample = pb.prepare_split(
            split,
            revision=spec.revision,
            seed=config.seed,
            sample_size=spec.sample_size,
            stratify=stratify,
            path=args.data_file,
        )
        stem = run.artifacts_dir / f"processbench_{split}"
        Path(f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        pb.write_jsonl(Path(f"{stem}_sample.jsonl"), (e.to_record() for e in sample))
        report = pb.format_report(manifest)
        Path(f"{stem}_report.md").write_text(report, encoding="utf-8")
        print(report)

        passed = pb.manifest_passed(manifest)
        all_passed &= passed
        summaries[split] = {
            "passed": passed,
            "sha256": manifest["dataset"]["sha256"],
            "num_valid": manifest["validation"]["num_valid"],
            "num_invalid": manifest["validation"]["num_invalid"],
            "sample_size": manifest["sampling"]["actual_size"],
        }

    run.finish(status="completed" if all_passed else "validation_failed", data_preparation=summaries)
    print(f"{'PASSED' if all_passed else 'FAILED'}: artifacts in {run.artifacts_dir}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
