# Experiment 2 — DCoT Reasoning Refinement

**Status:** not started (repository foundation only).

**Question.** Does fine-tuning with Divergent Chains of Thought (DCoT) — generating
several reasoning chains in a single inference pass — improve a small model's
reasoning and enable self-refinement? We reproduce the released Phi-1.5 DCoT
adapter's results, then study how the chains differ and when refinement helps.
Full scope: [`docs/RESEARCH_PLAN.md`](../../docs/RESEARCH_PLAN.md#experiment-2--dcot-reasoning-refinement).

| Component | Value |
|---|---|
| Config | [`configs/02_dcot_refinement.yaml`](../../configs/02_dcot_refinement.yaml) |
| Base model | `microsoft/phi-1_5` |
| Adapter | released DCoT adapter — ID to be verified |
| Evaluation data | `openai/gsm8k` placeholder; full suite to be confirmed from the paper |
| Outputs | `results/02_dcot_refinement/<run_id>/` (git-ignored) |

## Running

```bash
python experiments/02_dcot_refinement/run.py --dry-run
python experiments/02_dcot_refinement/run.py --set datasets.gsm8k.sample_size=20
```

## Before the first reported run

- [ ] Verify and fill the DCoT adapter ID; confirm prompt format and decoding settings from the released code.
- [ ] Confirm the evaluation datasets and splits used in the paper; add them to the config.
- [ ] Pin `revision` for the base model, adapter, and datasets.
- [ ] Record the reference numbers being reproduced in `docs/EXPERIMENT_LOG.md`.

Progress, decisions and results are logged in [`docs/EXPERIMENT_LOG.md`](../../docs/EXPERIMENT_LOG.md).
