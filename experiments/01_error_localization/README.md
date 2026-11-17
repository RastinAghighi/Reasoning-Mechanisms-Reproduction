# Experiment 1 — First-Error Localization

**Status:** not started (repository foundation only).

**Question.** Given a step-by-step math solution, can we identify the *first*
erroneous step? We reproduce a process reward model (SPARE-PRM) on
ProcessBench, then test whether token-level uncertainty (entropy, varentropy)
from Qwen2.5-Math-7B-Instruct localizes the first error without any
process supervision. Full scope: [`docs/RESEARCH_PLAN.md`](../../docs/RESEARCH_PLAN.md#experiment-1--first-error-localization).

| Component | Value |
|---|---|
| Config | [`configs/01_error_localization.yaml`](../../configs/01_error_localization.yaml) |
| Benchmark | `Qwen/ProcessBench` (gsm8k, math, olympiadbench, omnimath) |
| PRM (reproduction) | SPARE-PRM — checkpoint ID to be verified |
| Policy model (extension) | `Qwen/Qwen2.5-Math-7B-Instruct` |
| Outputs | `results/01_error_localization/<run_id>/` (git-ignored) |

## Running

```bash
# Inspect the resolved config without touching disk
python experiments/01_error_localization/run.py --dry-run

# Small smoke run with an override
python experiments/01_error_localization/run.py --set datasets.processbench.sample_size=20
```

## Before the first reported run

- [ ] Verify and fill the SPARE-PRM checkpoint ID.
- [ ] Pin `revision` for every model and dataset.
- [ ] Record the reference numbers being reproduced (paper table, metric definition) in `docs/EXPERIMENT_LOG.md`.

Progress, decisions and results are logged in [`docs/EXPERIMENT_LOG.md`](../../docs/EXPERIMENT_LOG.md).
