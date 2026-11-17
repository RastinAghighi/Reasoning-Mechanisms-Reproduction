# Experiment Log

Chronological record of implementation decisions, failures, fixes, and results.
Newest entries at the bottom. Each entry should be short and point to the run
directory (or `results/summaries/` file) that backs it.

**Entry template**

```markdown
### YYYY-MM-DD — <short title>
- **Experiment:** 01 | 02 | infra
- **Type:** decision | failure | fix | result
- **Run:** results/<experiment>/<run_id> (or n/a)
- **Details:** what was done and why.
- **Outcome / next step:**
```

For results, state explicitly whether a number is a **reproduction**
(and the paper value it is compared against) or an **extension**.

---

### 2026-11-17 — Repository foundation
- **Experiment:** infra
- **Type:** decision
- **Run:** n/a
- **Details:** Set up `src/qmind_reasoning` (YAML config with `extends` and
  `--set` overrides, strict schema validation, run directories with
  provenance metadata, seeding), per-experiment entry points and READMEs,
  and lightweight tests. Heavy ML dependencies are an optional extra
  (`.[experiments]`) so that the foundation installs and tests without them.
- **Outcome / next step:** Experiments not started. Open items before
  Experiment 1: verify SPARE-PRM checkpoint ID, pin model/dataset revisions,
  record the target paper numbers.

### 2026-11-17 — Exp 1 / Step 1: ProcessBench label semantics verified
- **Experiment:** 01
- **Type:** decision
- **Run:** n/a
- **Details:** Steps are indexed from 0. `label` is the index of the earliest
  erroneous step, and `-1` means no error. Three sources agree: paper Fig. 2,
  official `run_eval.py` (`QwenLM/ProcessBench@e802463`, `enumerate` → `<paragraph_0>`;
  exact-match scoring; split on `label == -1`), and the dataset-card example (`gsm8k-0`, label 1).
  The ground truth is `label`, not `final_answer_correct`.
- **Outcome / next step:** Encoded in `validate_split`. Labels outside `[-1, len(steps)-1]` are rejected.

### 2026-11-17 — Exp 1 / Step 1: data loader and sampler design
- **Experiment:** 01
- **Type:** decision
- **Run:** n/a
- **Details:** Pinned `Qwen/ProcessBench@3bdcd5371ed567559a78f559c01c13a6deee7604`.
  The split JSON is downloaded directly with urllib, so no `datasets` dependency
  is needed, and its SHA-256 is recorded. Sampling is stratified by correct/error
  and ranked by `sha256(seed:id)`, so it is deterministic across machines and
  nested (smoke ⊂ dev ⊂ full). Profiles have 20 / 100 / 400 examples.
- **Outcome / next step:** —

### 2026-11-17 — Exp 1 / Step 1: Windows console encoding failure
- **Experiment:** 01
- **Type:** failure → fix
- **Run:** n/a (failed run dirs deleted)
- **Details:** The first `prepare_data.py` runs crashed with `UnicodeEncodeError`
  when printing "≥" and en-dashes to a cp1252 console.
- **Outcome / next step:** The generated report is now ASCII-only.

### 2026-11-17 — Exp 1 / Step 1: GSM8K validation results
- **Experiment:** 01
- **Type:** result (data validation)
- **Run:** results/01_error_localization/20261117-190045_s42_b04bc521537d (smoke),
  …/20261117-190046_s42_f028ebb6b214 (dev), …/20261117-190047_s42_889ed1f83d30 (full)
- **Details:** File SHA-256 `fb1c59df…f61349`. 400/400 records valid, 0 warnings,
  and IDs are sequential `gsm8k-0..399`. 193 correct / 207 erroneous, with 7 erroneous
  samples whose final answer is correct; all three match paper Table 1. Steps per
  solution: 2–16 (mean 5.2, median 5). First-error index: 0–7 (mean 1.86,
  median 2), with histogram {0:37, 1:61, 2:50, 3:31, 4:13, 5:12, 6:2, 7:1}. In 16
  samples the error is at the last step. Smoke sample = 10/10 correct/error, dev =
  48/52. Smoke ⊂ dev ⊂ full was confirmed, and the sampled records equal the source records.
- **Outcome / next step:** Data is ready. Next: SPARE-PRM scoring (waiting for instruction).

### 2026-11-17 — Exp 1 / Step 2: SPARE-PRM inference recipe verified from sources
- **Experiment:** 01
- **Type:** decision
- **Run:** n/a
- **Details:** Pinned checkpoint `UKPLab/Qwen2.5-3b-spare-prm-math@a4f77040…` and code
  `UKPLab/aaai2026-spare-prm@d1076e0…`. The format is: system = training instruction
  (verbatim), user = problem, assistant = `"Let's think step by step.\n"` + `step ки\n`
  for each step. The score is `softmax([logit(-), logit(+)])[1]` at the tag token.
  Finding: in Qwen `" ки"` is two tokens (`" к"`, `"и"`), so the tag ID is 1802 (`"и"`),
  as given by the card's `encode(" ки")[-1]`. With the real tokenizer, all 400 GSM8K
  examples align and are 239–1,375 tokens long.
  Ambiguity: the card example includes the `"Let's think step by step."` prefix, but the
  training config sets `reasoning_trigger: ""`. We keep the card's version; it is configurable.
- **Outcome / next step:** See methodology §7.

### 2026-11-17 — Exp 1 / Step 2: SPARE publishes no GSM8K ProcessBench result
- **Experiment:** 01
- **Type:** decision needed
- **Run:** n/a
- **Details:** SPARE Table 3 reports ProcessBench for MATH / OlympiadBench / Omni-MATH only.
  For Qwen2.5-3B, error/correct/F1 are 16.0/89.2/27.1, 11.1/85.0/19.6 and 14.0/83.8/23.9.
  This follows the ProcessBench protocol (§4.1), which selects the PRM threshold by best F1
  *on GSM8K*. So our GSM8K dev/held-out results have no paper counterpart. A strict
  reproduction of Table 3 would require scoring the MATH split (1,000 examples), with the
  threshold chosen on GSM8K.
- **Outcome / next step:** To be decided by the team before the full run.

### 2026-11-17 — Exp 1 / Step 2: scoring module implemented (not run)
- **Experiment:** 01
- **Type:** decision
- **Run:** results/01_error_localization/20261117-192240_s42_5c348768f0a7 (smoke, tokenize-only),
  …/20261117-192249_s42_a792c4f1463f (full, tokenize-only)
- **Details:** Added the `StepScorer` interface (named per-step signals, ready for entropy
  and varentropy) and `SparePRMScorer`. The scorer applies the LM head only at tag positions,
  right-pads batches, records errors per example, and retries a failed batch one example at a time.
  Scores are written to resumable JSONL. The dev/held-out partition is 100/300 and identical to
  Step 1's dev sample; `load_score_records` blocks held-out labels unless `final_evaluation=True`.
  The tokenizer-only preflight passed (20/20 and 400/400 aligned). No model weights were downloaded.
- **Outcome / next step:** Awaiting permission to download weights and run the smoke run,
  starting with the model-card reference check.
