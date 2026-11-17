"""Resumable JSONL score files and partition-safe loading.

One line per example::

    {"id", "split", "partition", "label", "num_steps", "scorer", "signals",
     "num_tokens", "error", "model_id", "model_revision"}

Lines are appended and flushed after every batch, so an interrupted run can
be resumed: completed IDs are skipped, and a torn final line (crash mid-write)
is discarded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from qmind_reasoning.data.processbench import DEV, HELDOUT, ProcessBenchExample
from qmind_reasoning.scoring.base import StepScores


class HeldOutAccessError(PermissionError):
    """Raised when held-out records are requested outside final evaluation."""


def make_record(
    result: StepScores,
    example: ProcessBenchExample,
    split: str,
    partition: str,
    scorer: str,
    model_id: str,
    model_revision: str,
) -> dict[str, Any]:
    if result.id != example.id:
        raise ValueError(f"result {result.id!r} does not belong to example {example.id!r}")
    return {
        "id": example.id,
        "split": split,
        "partition": partition,
        "label": example.label,
        "num_steps": example.num_steps,
        "scorer": scorer,
        "signals": result.signals,
        "num_tokens": result.num_tokens,
        "error": result.error,
        "model_id": model_id,
        "model_revision": model_revision,
    }


class ScoreWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._repair_tail()

    def _repair_tail(self) -> None:
        """Drop a partially written last line, if any."""
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        if not data or data.endswith(b"\n"):
            return
        cut = data.rfind(b"\n") + 1  # 0 if there is no complete line
        self.path.write_bytes(data[:cut])

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def completed_ids(self, include_errors: bool = True) -> set[str]:
        """IDs already written. With ``include_errors=False``, failed IDs are retried."""
        return {r["id"] for r in self.records() if include_errors or r["error"] is None}

    def drop_errors(self) -> int:
        """Remove failed records so they can be re-scored. Returns how many were removed."""
        records = self.records()
        kept = [r for r in records if r["error"] is None]
        self.path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
        return len(records) - len(kept)

    def append(self, records: Iterable[Mapping[str, Any]]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def load_score_records(
    path: Path,
    partition: str,
    *,
    final_evaluation: bool = False,
    include_errors: bool = False,
) -> list[dict[str, Any]]:
    """Load scored records for one partition.

    Held-out records carry labels, so reading them requires an explicit
    ``final_evaluation=True``. Threshold or design tuning must only ever call
    this with ``partition="dev"``.
    """
    if partition not in (DEV, HELDOUT):
        raise ValueError(f"partition must be {DEV!r} or {HELDOUT!r}")
    if partition == HELDOUT and not final_evaluation:
        raise HeldOutAccessError(
            "held-out labels are reserved for final evaluation; pass final_evaluation=True "
            "only after all thresholds and design choices are frozen on dev"
        )
    records = ScoreWriter(path).records()
    return [r for r in records if r["partition"] == partition and (include_errors or r["error"] is None)]
