# Research Plan

This repository hosts two QMIND studies on the mechanisms of LLM reasoning.
Each has a **reproduction** part (re-running a published method and checking
that we recover its reported results) and an **extension** part (our own
questions, built on the reproduced setup). The two must never be blurred:
any number we report is labelled as one or the other.

| | Reproduction | Extension |
|---|---|---|
| Goal | Recover published numbers with released artifacts | Answer a new question |
| Success | Within an agreed tolerance of the paper | A defensible finding, positive or negative |
| Deviations from paper | Logged as failures/fixes | Are the point |
| Reported as | "Reproduced: X vs paper Y" | "Ours" |

Items marked **TBD** are open and must be resolved (and logged in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)) before the corresponding run.

---

## Experiment 1 — First-Error Localization

**Research question.** Given a multi-step math solution, how well can we
locate the *earliest* incorrect step — with a trained process reward model,
and with training-free uncertainty signals from the generating model?

### 1A. Reproduction: SPARE-PRM on ProcessBench

- **Benchmark.** ProcessBench (`Qwen/ProcessBench`): solutions annotated with
  the index of the first erroneous step, or −1 if fully correct. Subsets:
  GSM8K, MATH, OlympiadBench, Omni-MATH.
- **Method.** Score each step with the released SPARE-PRM checkpoint; the
  predicted first error is the first step whose score falls below a threshold
  (threshold selection procedure: **TBD**, follow the paper).
- **Metric.** ProcessBench F1: harmonic mean of exact-match accuracy on
  erroneous samples and on correct samples, per subset and averaged.
- **Target.** The SPARE-PRM ProcessBench numbers reported in the paper
  (exact table and tolerance: **TBD**, recorded in the log before running).

### 1B. Extension: entropy and varentropy as localization signals

- **Model.** `Qwen/Qwen2.5-Math-7B-Instruct`, teacher-forced over each
  ProcessBench solution (no generation).
- **Signals.** Per token, with next-token distribution *p*:
  entropy H = −Σ p log p; varentropy V = Σ p (log p + H)².
  Aggregated per step (mean / max / other: **TBD**).
- **Questions.**
  1. Do steps containing the first error have higher entropy/varentropy than
     preceding correct steps?
  2. How does a training-free localizer (e.g. first step exceeding a
     threshold) compare to SPARE-PRM on ProcessBench F1?
  3. Are the signals complementary to the PRM (e.g. in combination)?
- **Controls.** Step length, position in solution, and subset difficulty, since
  each can confound raw uncertainty.

## Experiment 2 — DCoT Reasoning Refinement

**Research question.** Does training a small model to produce several
divergent reasoning chains in a single pass (DCoT) improve its reasoning,
and does it enable self-refinement across chains?

### 2A. Reproduction: Phi-1.5 with the released DCoT adapter

- **Model.** `microsoft/phi-1_5` + released DCoT adapter (ID: **TBD**, verify).
- **Method.** Prompt for *k* chains followed by a final answer, k ∈ {1, 2, 3, 4};
  compare with the CoT baseline reported in the paper.
- **Data and metric.** The paper's evaluation datasets and accuracy metric
  (exact list: **TBD**; GSM8K is a placeholder in the config).
- **Target.** The Phi-1.5 rows of the paper's results tables, including the
  DCoT@1 → DCoT@2 gain attributed to self-correction.

### 2B. Extension (candidates — to be agreed before implementation)

- Characterize *where* later chains diverge from earlier ones and whether the
  divergence point coincides with an error in the earlier chain.
- Link to Experiment 1: do entropy/varentropy (or a PRM) at the divergence
  point predict whether refinement fixes the answer?
- Robustness of the refinement gain across seeds, prompts, and sample sizes.

---

## Reproducibility standards

- Every run is driven by a config in `configs/` and writes the resolved
  config plus provenance (git commit, package versions, seed, overrides) to
  its run directory under `results/`.
- Model and dataset `revision`s are pinned before any reported run; runs
  record any that are unpinned.
- Smoke runs use `sample_size`; reported runs use full splits unless stated.
- Large artifacts (weights, caches, datasets, raw outputs) are never committed.
  Curated summaries that back a claim go in `results/summaries/`.
