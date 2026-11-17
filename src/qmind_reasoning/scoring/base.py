"""Common interface for step-level scorers.

Every scorer maps a ProcessBench example to one value per step for each of
its *signals*. Signal lists are indexed exactly like ``example.steps`` (0-based,
same as the ProcessBench ``label``), so ``signals[name][i]`` always refers to
step ``i``. A scorer that cannot guarantee this for an example must return an
error for it instead of a misaligned list.

Examples of signals:
    SPARE-PRM:            {"prm_correct_prob": [...]}
    entropy (planned):    {"entropy_mean": [...], "varentropy_mean": [...], ...}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from qmind_reasoning.data.processbench import ProcessBenchExample


class AlignmentError(ValueError):
    """Model positions cannot be mapped one-to-one onto ProcessBench steps."""


@dataclass
class StepScores:
    id: str
    num_steps: int
    signals: dict[str, list[float]] = field(default_factory=dict)
    num_tokens: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def check(self, expected_signals: Sequence[str]) -> "StepScores":
        """Enforce the alignment contract; converts violations into an error result."""
        if self.error is not None:
            self.signals = {}
            return self
        missing = [s for s in expected_signals if s not in self.signals]
        wrong = {k: len(v) for k, v in self.signals.items() if len(v) != self.num_steps}
        if missing or wrong:
            self.error = f"AlignmentError: missing signals {missing}, wrong lengths {wrong} (num_steps={self.num_steps})"
            self.signals = {}
        return self

    @classmethod
    def failure(cls, example: ProcessBenchExample, error: BaseException | str, num_tokens: int | None = None):
        message = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        return cls(id=example.id, num_steps=example.num_steps, num_tokens=num_tokens, error=message)


class StepScorer(ABC):
    """Base class for scorers. Subclasses must be safe to call repeatedly."""

    name: str = "abstract"
    signal_names: tuple[str, ...] = ()

    @abstractmethod
    def score(self, examples: Sequence[ProcessBenchExample]) -> list[StepScores]:
        """Return one result per input example, in input order. Never raise for a
        per-example problem; record it in ``StepScores.error`` instead."""

    @abstractmethod
    def runtime_info(self) -> dict[str, Any]:
        """Everything needed to reproduce the scores (model, revision, format, device...)."""
