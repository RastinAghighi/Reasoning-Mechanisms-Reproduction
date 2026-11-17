# Experiment 1 — First-Error Localization

**Status:** Step 1 (data) done. Step 2 (SPARE-PRM scoring) is implemented, and the tokenizer preflight passes. Model inference has not run yet.

**Question.** Given a step-by-step math solution, can we identify the *first*
erroneous step? We reproduce a process reward model (SPARE-PRM) on
ProcessBench, then test whether token-level uncertainty (entropy, varentropy)
from Qwen2.5-Math-7B-Instruct localizes the first error without any
process supervision. Full scope: [`docs/RESEARCH_PLAN.md`](../../docs/RESEARCH_PLAN.md#experiment-1--first-error-localization).
Data handling and evaluation design: [`docs/EXPERIMENT_01_METHODOLOGY.md`](../../docs/EXPERIMENT_01_METHODOLOGY.md).

| Component | Value |
|---|---|
| Configs | [`01_error_localization.yaml`](../../configs/01_error_localization.yaml) (full, 400) · [`_dev`](../../configs/01_error_localization_dev.yaml) (100) · [`_smoke`](../../configs/01_error_localization_smoke.yaml) (20) |
| Benchmark | `Qwen/ProcessBench`, split `gsm8k`, revision `3bdcd53` (pinned) |
| PRM (reproduction) | `UKPLab/Qwen2.5-3b-spare-prm-math` @ `a4f7704` (pinned) |
| Policy model (extension) | `Qwen/Qwen2.5-Math-7B-Instruct` |
| Outputs | `results/01_error_localization/<run_id>/` (git-ignored) |

## Steps

| Step | Script | Status |
|---|---|---|
| 1. Data preparation and validation | `prepare_data.py` | done |
| 2. SPARE-PRM scoring (reproduction) | `score_spare.py` | implemented, not run |
| 3. Entropy/varentropy scoring (extension) | — | not started |
| 4. Evaluation and analysis | — | not started |

## Running

```bash
# Step 1: load, validate, sample, and write the manifest (no model weights)
python experiments/01_error_localization/prepare_data.py --config configs/01_error_localization_smoke.yaml
python experiments/01_error_localization/prepare_data.py --config configs/01_error_localization_dev.yaml
python experiments/01_error_localization/prepare_data.py   # full split

# Resolve and print the config without writing anything
python experiments/01_error_localization/prepare_data.py --dry-run

# Step 2: SPARE-PRM step scores
python experiments/01_error_localization/score_spare.py --config configs/01_error_localization_smoke.yaml --tokenize-only  # no weights
python experiments/01_error_localization/score_spare.py --config configs/01_error_localization_smoke.yaml                  # 20 examples
python experiments/01_error_localization/score_spare.py --resume results/01_error_localization/<run_id> [--retry-errors]

# Opt-in tests
QMIND_INTEGRATION=1 pytest -m integration   # real data + tokenizer (~16 MB)
QMIND_MODEL_TESTS=1 pytest -m model         # real weights: reproduces the model-card example scores
```

The dev set is the 100-example dev profile, and the other 300 examples are held out.
Tune only on dev; see [methodology §8](../../docs/EXPERIMENT_01_METHODOLOGY.md#8-dev--held-out-partition).

`prepare_data.py` exits non-zero if validation fails or the class counts
don't match the paper.

## Before the first reported model run

- [x] Pin the ProcessBench revision.
- [x] Verify and pin the SPARE-PRM checkpoint.
- [x] Record the published SPARE numbers and threshold protocol (methodology §7.4). Note: none are published for GSM8K.
- [ ] Reproduce the model-card example scores (`QMIND_MODEL_TESTS=1 pytest -m model`).
- [ ] Decide whether to add the MATH split for a strict Table 3 reproduction.
- [ ] Pin the Qwen2.5-Math-7B-Instruct revision (extension).

Progress, decisions and results are logged in [`docs/EXPERIMENT_LOG.md`](../../docs/EXPERIMENT_LOG.md).
