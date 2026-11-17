# Experiment 1 Methodology: Dataset Handling and Evaluation Design

Scope: how ProcessBench data enters the pipeline (§1–5), how first-error
localization will be evaluated (§6), how SPARE-PRM step scores are produced
(§7), and the dev/held-out split (§8). Entropy/varentropy scoring, threshold
tuning, and plots are **not implemented yet**.

## 1. Data source

| Item | Value |
|---|---|
| Dataset | [`Qwen/ProcessBench`](https://huggingface.co/datasets/Qwen/ProcessBench) |
| Split | `gsm8k` (400 examples). Other splits: `math`, `olympiadbench`, `omnimath` (1,000 each) |
| Revision | `3bdcd5371ed567559a78f559c01c13a6deee7604` (Hub commit, 2024-12-27) |
| File | `gsm8k.json` (JSON array), SHA-256 `fb1c59dfd1e83e1c5a48dbe598db27e4bc0b8bc524a09445f2d4b69178f61349` |
| License | Apache-2.0 |

The split file is fetched straight from the pinned revision, with no
`datasets` dependency, and cached under `.cache/datasets/` (git-ignored).
Its SHA-256 is recorded in every manifest. The integration test asserts that
SHA-256, so any change to the source is caught.

## 2. Schema and label semantics

Each record has these fields:

| Field | Type | Meaning |
|---|---|---|
| `id` | str | Stable sample ID, `"<split>-<n>"` |
| `generator` | str | Model that wrote the solution |
| `problem` | str | Math problem |
| `steps` | list[str] | Solution split into paragraphs |
| `final_answer_correct` | bool | Whether the final answer matches the reference |
| `label` | int | 0-based index of the **earliest** erroneous step, or `-1` if no step is wrong |

**How the semantics were verified** (all three sources agree):

1. **Paper** (arXiv:2412.06559), Figure 2: "The label 2 denotes that the
   earliest error occurs in the 2nd step (indexed from 0). For test cases with
   no errors, the labels are −1."
2. **Official evaluation code** (`QwenLM/ProcessBench@e802463`,
   `code/run_eval.py`): steps are tagged `<paragraph_{sdx}>` with
   `sdx` from `enumerate(steps)`, which starts at 0. The prediction must equal `label`
   exactly, and samples are split into `label != -1` ("error") and `label == -1` ("correct").
   The critique prompt says the paragraphs are "indexed from 0" and that −1 means "not found".
3. **Data and paper counts**. The pinned file reproduces Table 1: 207 erroneous,
   193 correct, and 7 erroneous samples whose final answer is still correct
   ((200 − 193)/200 = 3.5% in the paper). Record `gsm8k-0` matches the
   dataset-card example (label 1, `final_answer_correct: false`).

Consequences for our code:

- The ground truth for "has an error" is `label != -1`, **not**
  `final_answer_correct`. Seven GSM8K samples reach a correct answer through a
  flawed process.
- The valid label range is `[-1, len(steps) - 1]`. A label equal to `len(steps)` is
  an off-by-one error and is rejected.
- Step text is used exactly as stored. Prompt formats that join or tag steps are a
  model-side decision and will be documented with that step.

## 3. Validation

`qmind_reasoning.data.processbench.validate_split` checks every record.

**Errors.** The record is excluded from evaluation and listed in the manifest:
- a required field is missing or has the wrong type (`bool` is not accepted as `int`);
- `steps` is empty, or a step is not a string;
- `id` is empty or duplicated (the first occurrence is kept);
- `label` is outside `[-1, len(steps) - 1]`.

**Warnings.** The record is kept and flagged:
- the `id` prefix does not match the split;
- a step or the problem is blank;
- `label == -1` but `final_answer_correct` is false (a correct process with a wrong answer).

**Split-level checks:**
- the record count matches the official size (400 for GSM8K);
- the class counts match the paper's Table 1 (`compare_to_paper`).

A preparation run passes only if there are no invalid records, no split-level
issues, and the paper counts match.

**Result on the pinned GSM8K split:** 400/400 valid, 0 warnings, and the paper
counts match. See [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

## 4. Sampling

| Profile | Config | Size | Use |
|---|---|---|---|
| smoke | `configs/01_error_localization_smoke.yaml` | 20 | Pipeline checks only |
| dev | `configs/01_error_localization_dev.yaml` | 100 | Method iteration, threshold exploration |
| full | `configs/01_error_localization.yaml` | 400 | Reported results |

Method (`stratified-sha256-v1`):

1. Valid examples are split into two strata: correct (`label == -1`) and error.
2. The sample size is divided between the strata in proportion to their size
   (largest-remainder rounding). This preserves the class ratio, which matters
   because ProcessBench F1 needs both classes.
3. Within each stratum, examples are ranked by `sha256(f"{seed}:{id}")`, and the
   lowest-ranked ones are taken.
4. The selected examples are returned in their original file order.

Properties:
- **Deterministic and portable.** The sample does not depend on Python's
  `random` implementation, the platform, or the input order.
- **Nested.** For a fixed seed, smoke ⊂ dev ⊂ full. This was verified on the
  real data with seed 42.
- **Annotations unchanged.** Sampling only selects IDs. The saved sample
  records were checked to be byte-for-byte identical (as parsed JSON) to the source.

The smoke sample (seed 42) has 10 correct and 10 erroneous examples. The dev
sample has 48 correct and 52 erroneous.

## 5. Artifacts and manifests

`experiments/01_error_localization/prepare_data.py` writes to
`results/01_error_localization/<run_id>/artifacts/`:

- `processbench_gsm8k_manifest.json`: dataset ID, revision, source URL,
  SHA-256, sampling method/seed/size, ordered `sample_ids`, full validation
  results, a paper-reference check, and statistics for the full split and the sample;
- `processbench_gsm8k_sample.jsonl`: the selected records, verbatim;
- `processbench_gsm8k_report.md`: a readable summary.

The run's `metadata.json` adds the git state, package versions, config
fingerprint, and pass/fail status. Downstream steps must read examples from
a manifest/sample file instead of re-sampling.

## 6. Evaluation design (to be implemented)

Following the official ProcessBench evaluation:

- **Prediction.** One integer per example: the predicted index of the earliest
  erroneous step, or −1. A prediction is correct only if it **exactly** equals `label`.
- **Error accuracy** = accuracy over `label != -1`. **Correct accuracy** = accuracy
  over `label == -1`.
- **F1** = harmonic mean of the two accuracies (in %), as in `run_eval.py`. We
  define F1 = 0 when both accuracies are 0; the official code would divide by zero.
- **PRM predictions** (reproduction). The predicted first error is the first step
  whose PRM score falls below a threshold, and −1 if none does. The official
  ProcessBench protocol (paper §4.1) picks the threshold with the highest F1
  **on the GSM8K subset** and reports the other subsets. In our GSM8K-only
  design, the threshold is tuned on the 100 dev examples and reported on the
  300 held-out examples (§8). See §7.4 for how this relates to the published SPARE numbers.
- **Uncertainty predictions** (extension). Analogous thresholding on step-level
  entropy/varentropy. Thresholds and aggregation choices are selected on the
  dev profile and then reported on the full split. Because dev ⊂ full,
  full-split results will also be reported on the 300 held-out examples
  (full − dev) to avoid optimistic bias.
- **Uncertainty reporting.** Bootstrap confidence intervals over examples. Results
  on 20 examples are never treated as evidence.

## 7. SPARE-PRM scoring (implemented; not yet run)

### 7.1 Sources (pinned)

| Item | Value |
|---|---|
| Checkpoint | [`UKPLab/Qwen2.5-3b-spare-prm-math`](https://huggingface.co/UKPLab/Qwen2.5-3b-spare-prm-math) @ `a4f77040cadc6a0f5f7758ec42d3f999376aef4a` (2026-03-31) |
| Architecture | `Qwen2ForCausalLM`, 3.09B params, stored in bf16 (6.17 GB, 2 safetensors shards), fine-tuned from Qwen2.5-3B-Instruct on MATH generations |
| Code | [`UKPLab/aaai2026-spare-prm`](https://github.com/UKPLab/aaai2026-spare-prm) @ `d1076e02de3a326bca2a2c3dc92c8821c078175b`: `infer_rm.py`, `data/rm_utils.py`, `data/utils.py`, `prompting/utils.py`, `models/RewardModel/token_classifier_rmutils.py`, `config/infer-rm-config.yaml`, `config/train-tc-rm-config.yaml` |
| Paper | Rizvi, Zhu & Gurevych, AAAI 2026 (arXiv:2506.15498), Table 3 |

### 7.2 Input format (`qmind_reasoning.scoring.spare_prm.SpareFormat`)

```
system:    <fixed training instruction, verbatim from the model card>
user:      <ProcessBench problem, verbatim>
assistant: "Let's think step by step.\n" + "".join(f"{step.strip()} ки\n" for step in steps)
```

The messages are rendered with the checkpoint's own chat template (no generation prompt) and tokenized without extra special tokens.

How each choice was verified:

- **Instruction and roles** (system/user/assistant): model card code, copied verbatim.
- **Step tag `" ки"` followed by `"\n"`:** the training config `step_tag: " ки\n"` and the model-card examples.
- **`"Let's think step by step.\n"` prefix:** taken from the model-card example. The training config sets `reasoning_trigger: ""`, but the generations being scored there may already contain it. This is the one ambiguity: the prefix is configurable (`params.spare.reasoning_trigger`) and can be ablated on dev.
- **`strip()` on each step:** the training regex `\n*\[\d+\]\s*` consumed whitespace around step markers. This affects 16 GSM8K steps; the text inside steps is kept.
- **Equivalence to the authors' code:** rendering and then tokenizing gives the same IDs as the card's `apply_chat_template(tokenize=True)`. This is tested with the real tokenizer on the card example.
- **Trailing text:** SPARE's `form_response_text` calls `.strip()`, while the card keeps the final `"\n"`. Because the model is causal, nothing after the last tag can change any score.

### 7.3 Score extraction and step alignment

- **Step-tag token:** `tokenizer.encode(" ки")[-1]`, as in the model card. In this Qwen tokenizer `" ки"` is **two tokens**, `" к"` (7665) and `"и"` (1802), so scores are read at `"и"`. The training code's `encode(step_tag)[1]` gives the same ID, because Qwen adds no BOS.
- **Targets:** `"-"` = 12 (incorrect) and `"+"` = 10 (correct).
- **Score:** for each tag position, `softmax([logit("-"), logit("+")])[1]`, computed in float32.
- **Alignment:** the *i*-th tag token is ProcessBench step *i* (0-indexed, the same as `label`). If the number of tag tokens doesn't equal the number of steps, for example because a stray `"и"` token appears in Cyrillic text, the example is recorded as an `AlignmentError` and gets **no** scores. It is never partially scored. On the pinned GSM8K split, 400/400 examples align (239–1,375 tokens; training `max_length` was 2,048).
- **Implementation:** the trunk runs once, and the LM head is applied only at tag positions. These are the same dot products as the full-logit computation, at a fraction of the memory. Tests on a random tiny Qwen2 model check that the result matches the card formula `model(ids).logits[..., [-, +]].softmax(-1)[..., 1][ids == tag]` to within 1e-6.
- **Padding:** batches are right-padded, as in the authors' code. With causal attention, padding cannot affect tag positions. Batched and unbatched scores agree to within 1e-5 (tested).
- **Failure handling:** errors are recorded per example (alignment, `max_tokens` exceeded, forward-pass errors). Inputs are never truncated. If a batch fails, for example with out-of-memory, it is retried one example at a time.

### 7.4 What can and cannot be compared with the paper

- The SPARE paper's ProcessBench results (Table 3) cover **only MATH, OlympiadBench, and Omni-MATH**, following the ProcessBench protocol, which uses GSM8K to pick the threshold. For SPARE-Qwen2.5-3B, error / correct / F1 are: MATH 16.0 / 89.2 / 27.1; OlympiadBench 11.1 / 85.0 / 19.6; Omni-MATH 14.0 / 83.8 / 23.9.
- **There is no published GSM8K number to reproduce.** GSM8K results from this repository are therefore our own measurement with the released model and inference recipe. Reproducing Table 3 would need the MATH (and other) splits, scored with a threshold chosen on GSM8K. This is open for decision; see `EXPERIMENT_LOG.md`.
- **Implementation-level verification:** the model card prints example scores: `[0.8710, 0.9163, 0.9786]` (correct) and `[0.3292, 0.5288]` (incorrect). `tests/test_spare_integration.py::test_reproduces_model_card_scores` checks these (fp32 tolerance 1e-3, bf16 tolerance 3e-2). It needs the weights, so it has **not been run yet**.

### 7.5 Runtime and outputs

`experiments/01_error_localization/score_spare.py` writes the following to the run's `artifacts/`:

- `spare_gsm8k_scores.jsonl`: one line per example with `id, split, partition, label, num_steps, scorer, signals.prm_correct_prob[num_steps], num_tokens, error, model_id, model_revision`. It is appended and fsynced after every batch. `--resume <run_dir>` skips finished IDs and drops a half-written last line. `--retry-errors` re-scores failed examples.
- `spare_gsm8k_runtime.json`: one entry per session, recording model revision, dtype, device, batch size, format strings, token IDs, attention implementation, and timing.
- `processbench_gsm8k_manifest.json` and `processbench_gsm8k_partition.json`. On resume, the data hash and sample IDs must match the saved manifest, and the frozen config must match its fingerprint.
- `--tokenize-only` checks formatting and alignment with the tokenizer only, without loading weights.

Settings live in `params.spare`: `device` (`auto|cpu|cuda|cuda:N`), `batch_size`, `max_tokens`, `deterministic`, and `reasoning_trigger`. The dtype comes from `models.prm.dtype`. When `deterministic` is on, all RNGs are seeded, torch deterministic algorithms are enabled (warn-only), TF32 is disabled, and the model runs in eval mode with no sampling.

## 8. Dev / held-out partition

- **dev** = the 100-example dev-profile sample (seed 42, stratified). **held-out** = the remaining 300 valid examples. This is computed by `partition_dev_heldout` and saved in every scoring run, and it was checked to equal Step 1's dev sample. The smoke sample (20) lies inside dev.
- Every score record carries its `partition`. `load_score_records(path, "heldout")` raises `HeldOutAccessError` unless called with `final_evaluation=True`. All tuning (thresholds, step aggregation, prompt-format ablations) must use only `partition="dev"`.
- Held-out results are computed once, after the choices are frozen and logged in `EXPERIMENT_LOG.md`.
- Scoring held-out examples does not leak anything, because scores never use labels. Only *reading held-out labels* is restricted.
