"""Experiment 2 entry point: DCoT reasoning refinement with Phi-1.5.

Usage:
    python experiments/02_dcot_refinement/run.py --dry-run
    python experiments/02_dcot_refinement/run.py --set datasets.gsm8k.sample_size=20

The experiment pipeline is intentionally not implemented yet; this script
only resolves the config and initializes a run directory.
"""

from __future__ import annotations

import sys

from qmind_reasoning.cli import prepare
from qmind_reasoning.paths import CONFIGS_DIR

DEFAULT_CONFIG = CONFIGS_DIR / "02_dcot_refinement.yaml"


def main(argv: list[str] | None = None) -> int:
    _, run, _ = prepare(DEFAULT_CONFIG, __doc__.splitlines()[0], argv)
    if run is None:
        return 0
    run.finish(status="not_implemented")
    print("Pipeline not implemented yet (repository foundation only).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
