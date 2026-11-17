# Reasoning Mechanisms: Reproduction & Extensions

QMIND research codebase for two studies on how language models reason and
where they go wrong:

| # | Study | Reproduction | Extension |
|---|---|---|---|
| 1 | [First-error localization](experiments/01_error_localization/README.md) | SPARE-PRM on ProcessBench | Entropy / varentropy of Qwen2.5-Math-7B-Instruct as a training-free signal |
| 2 | [DCoT reasoning refinement](experiments/02_dcot_refinement/README.md) | Phi-1.5 with the released DCoT adapter | Analysis of when and how refinement across chains helps |

- **Research plan:** [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) — questions, methods, and the reproduction/extension boundary.
- **Experiment 1 methodology:** [`docs/EXPERIMENT_01_METHODOLOGY.md`](docs/EXPERIMENT_01_METHODOLOGY.md) — data handling and evaluation design.
- **Experiment log:** [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) — decisions, failures, fixes, results.

## Setup

Requires Python ≥ 3.10.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"            # foundation + tests (lightweight)
pip install -e ".[experiments]"    # torch, transformers, datasets, peft — only to run experiments
cp .env.example .env               # optional: set HF_HOME to keep the HF cache out of the repo
```

Install a CUDA-specific `torch` build first if your GPU requires it.

```bash
pytest                                                   # run tests
python experiments/01_error_localization/run.py --dry-run  # show resolved config
```

## Layout

```
configs/        YAML configs (base.yaml + one per experiment)
experiments/    one directory per study: README + run.py entry point
src/qmind_reasoning/   shared code: config, run directories, seeding, CLI
tests/          lightweight tests (no model or dataset downloads)
docs/           research plan and experiment log
results/        per-run outputs (git-ignored; curated summaries/ tracked)
```

## Reproducibility

Each run resolves its config (inheritance via `extends`, overrides via
`--set key=value`), seeds all RNGs, and writes `config.yaml` and
`metadata.json` (git commit, package versions, seed, overrides, unpinned
revisions) to `results/<experiment>/<run_id>/`. Model weights, Hugging Face
caches, datasets, and large outputs are never committed.
