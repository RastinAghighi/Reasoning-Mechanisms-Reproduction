"""Experiment 1, step 2: score ProcessBench steps with SPARE-PRM.

For each configured split: load and validate the pinned data, select the
profile's sample, assign dev/held-out partitions, then write

    artifacts/processbench_<split>_manifest.json    data provenance + sample IDs
    artifacts/processbench_<split>_partition.json   dev (100) vs held-out (300) IDs
    artifacts/spare_<split>_scores.jsonl            one line per example (resumable)
    artifacts/spare_<split>_runtime.json            model revision, format, device, ... per session
    artifacts/spare_<split>_tokenization.json       (--tokenize-only) alignment/length pre-flight

Usage:
    # Pre-flight without model weights (downloads the tokenizer only)
    python experiments/01_error_localization/score_spare.py --config configs/01_error_localization_smoke.yaml --tokenize-only
    # Scoring (downloads ~6.2 GB of weights on first use)
    python experiments/01_error_localization/score_spare.py --config configs/01_error_localization_smoke.yaml
    # Resume an interrupted run (optionally re-try failed examples)
    python experiments/01_error_localization/score_spare.py --resume results/01_error_localization/<run_id> [--retry-errors]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from qmind_reasoning.cli import build_parser, describe, start
from qmind_reasoning.config import ExperimentConfig
from qmind_reasoning.data import processbench as pb
from qmind_reasoning.paths import CONFIGS_DIR, PROJECT_ROOT
from qmind_reasoning.runs import RunContext, open_run
from qmind_reasoning.scoring.io import ScoreWriter, make_record
from qmind_reasoning.scoring.spare_prm import SparePRMScorer, SpareFormat, batched
from qmind_reasoning.seeding import set_seed

DEFAULT_CONFIG = CONFIGS_DIR / "01_error_localization.yaml"
DATASET_KEY, MODEL_KEY = "processbench", "prm"

# Keep Hub downloads in the git-ignored project cache unless the user chose a location.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def build_scorer(config: ExperimentConfig) -> SparePRMScorer:
    model, opts = config.models[MODEL_KEY], config.params.get("spare", {})
    fmt = SpareFormat(reasoning_trigger=opts.get("reasoning_trigger", SpareFormat.reasoning_trigger))
    return SparePRMScorer(
        model_id=model.id,
        revision=model.revision,
        dtype=model.dtype,
        device=opts.get("device", "auto"),
        batch_size=int(opts.get("batch_size", 1)),
        max_tokens=opts.get("max_tokens", 4096),
        fmt=fmt,
    )


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_split_data(config: ExperimentConfig, split: str, artifacts: Path):
    """Load/validate/sample/partition; on resume, verify against the saved manifest."""
    spec = config.datasets[DATASET_KEY]
    stratify = bool(config.params.get("stratify_sampling", True))
    records, source = pb.load_split(split, spec.revision)
    report = pb.validate_split(records, split)
    sample = pb.sample_examples(report.valid, spec.sample_size, config.seed, stratify)
    manifest = pb.build_manifest(source, report, sample, config.seed, spec.sample_size, stratify)
    if not pb.manifest_passed(manifest):
        raise SystemExit(f"data validation failed for {split}; run prepare_data.py for details")

    partition = pb.partition_dev_heldout(report.valid, int(config.params["dev_size"]), config.seed, stratify)
    manifest_path = artifacts / f"processbench_{split}_manifest.json"
    if manifest_path.exists():  # resume: the data must be byte-identical
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (saved["dataset"]["sha256"], saved["sample_ids"]) != (source.sha256, manifest["sample_ids"]):
            raise SystemExit("data or sample differs from the saved manifest; refusing to resume")
    else:
        _write_json(manifest_path, manifest)
        _write_json(
            artifacts / f"processbench_{split}_partition.json",
            {
                "rule": "dev = dev-profile sample (stratified-sha256-v1); heldout = remaining valid examples",
                "seed": config.seed,
                "dev_size": int(config.params["dev_size"]),
                "dev_ids": [i for i, p in partition.items() if p == pb.DEV],
                "heldout_ids": [i for i, p in partition.items() if p == pb.HELDOUT],
            },
        )
    return sample, partition


def tokenize_only(scorer: SparePRMScorer, sample, split: str, artifacts: Path) -> bool:
    rows, failures = [], []
    for example in sample:
        try:
            ids, positions = scorer.prepare(example)
            rows.append(len(ids))
        except Exception as exc:
            failures.append({"id": example.id, "error": f"{type(exc).__name__}: {exc}"})
    summary = {
        "split": split,
        "num_examples": len(sample),
        "num_aligned": len(rows),
        "failures": failures,
        "tokens": {"min": min(rows), "max": max(rows), "mean": round(sum(rows) / len(rows), 1)} if rows else None,
        "token_ids": vars(scorer.token_ids),
    }
    _write_json(artifacts / f"spare_{split}_tokenization.json", summary)
    print(f"[{split}] tokenization: {len(rows)}/{len(sample)} aligned, tokens={summary['tokens']}")
    for f in failures:
        print(f"  FAIL {f['id']}: {f['error']}")
    return not failures


def score_split(scorer: SparePRMScorer, sample, partition, split: str, artifacts: Path, retry_errors: bool) -> dict:
    writer = ScoreWriter(artifacts / f"spare_{split}_scores.jsonl")
    if retry_errors:
        print(f"[{split}] dropped {writer.drop_errors()} failed record(s) for retry")
    done = writer.completed_ids()
    pending = [e for e in sample if e.id not in done]
    print(f"[{split}] {len(done)} already scored, {len(pending)} pending")

    runtime_path = artifacts / f"spare_{split}_runtime.json"
    sessions = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else []
    session = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "pending": len(pending)}

    started = time.perf_counter()
    for chunk in batched(pending, scorer.batch_size):
        results = scorer.score(chunk)
        writer.append(
            make_record(r, e, split, partition[e.id], scorer.name, scorer.model_id, scorer.revision)
            for r, e in zip(results, chunk)
        )
        for r in results:
            status = "ok" if r.ok else f"ERROR {r.error}"
            print(f"  {r.id}: {r.num_steps} steps, {r.num_tokens} tokens, {status}")

    session.update(
        runtime=scorer.runtime_info() if pending else None,
        seconds=round(time.perf_counter() - started, 1),
        finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    _write_json(runtime_path, [*sessions, session])

    records = writer.records()
    errors = [r["id"] for r in records if r["error"] is not None]
    return {
        "scored": len(records),
        "expected": len(sample),
        "errors": errors,
        "by_partition": {p: sum(r["partition"] == p for r in records) for p in (pb.DEV, pb.HELDOUT)},
    }


def run_scoring(
    config: ExperimentConfig,
    run: RunContext,
    *,
    tokenize: bool = False,
    retry_errors: bool = False,
    scorer_factory: Callable[[ExperimentConfig], SparePRMScorer] = build_scorer,
) -> int:
    set_seed(config.seed, deterministic=bool(config.params.get("spare", {}).get("deterministic", True)))
    scorer = scorer_factory(config)
    summaries, ok = {}, True
    for split in config.datasets[DATASET_KEY].splits:
        sample, partition = prepare_split_data(config, split, run.artifacts_dir)
        if tokenize:
            ok &= tokenize_only(scorer, sample, split, run.artifacts_dir)
            continue
        summary = score_split(scorer, sample, partition, split, run.artifacts_dir, retry_errors)
        summaries[split] = summary
        ok &= summary["scored"] == summary["expected"] and not summary["errors"]
        print(f"[{split}] {summary['scored']}/{summary['expected']} scored, {len(summary['errors'])} error(s)")

    if tokenize:
        run.update_metadata(status="tokenization_checked" if ok else "tokenization_failed")
    else:
        run.finish(status="completed" if ok else "completed_with_errors", spare_scoring=summaries)
    return 0 if ok else 1


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resume", type=Path, help="continue an existing run directory")
    parser.add_argument("--retry-errors", action="store_true", help="re-score examples that previously failed")
    parser.add_argument(
        "--tokenize-only",
        action="store_true",
        help="check formatting/alignment with the tokenizer only (no model weights)",
    )


def main(argv: list[str] | None = None, scorer_factory=build_scorer) -> int:
    parser = build_parser(DEFAULT_CONFIG, __doc__.splitlines()[0])
    add_arguments(parser)
    args = parser.parse_args(argv)

    if args.resume:
        if args.overrides:
            parser.error("--set cannot be combined with --resume (the run's config is frozen)")
        run = open_run(args.resume)
        config = run.config
        describe(config, run.run_dir / "config.yaml")
        print(f"resuming   : {run.run_dir}")
    else:
        config, run = start(args)
        if run is None:
            return 0
    return run_scoring(
        config, run, tokenize=args.tokenize_only, retry_errors=args.retry_errors, scorer_factory=scorer_factory
    )


if __name__ == "__main__":
    sys.exit(main())
