"""SPARE-PRM step scoring, following the authors' released inference recipe.

Sources (pinned; see docs/EXPERIMENT_01_METHODOLOGY.md §7):
* Model card, UKPLab/Qwen2.5-3b-spare-prm-math @ a4f77040cadc6a0f5f7758ec42d3f999376aef4a
* Code, UKPLab/aaai2026-spare-prm @ d1076e02de3a326bca2a2c3dc92c8821c078175b
  (spare-prm/infer_rm.py, data/rm_utils.py, models/RewardModel/token_classifier_rmutils.py)

Recipe:
1. Chat messages: system = the fixed training instruction, user = problem,
   assistant = "Let's think step by step.\\n" + "".join(f"{step} ки\\n").
2. Render with the tokenizer's chat template (no generation prompt).
3. Step-tag token id = tokenizer.encode(" ки")[-1]. For this Qwen tokenizer,
   " ки" is two tokens (" к", "и"); the authors score at the last one.
4. At each tag position, take logits of ["-", "+"], softmax over the two, and
   keep P("+") as the step-correctness score.

Alignment: the i-th tag occurrence is ProcessBench step i. If the number of
tag tokens differs from the number of steps (e.g. a stray "и" token), the
example is reported as an error instead of being scored.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from qmind_reasoning.data.processbench import ProcessBenchExample
from qmind_reasoning.scoring.base import AlignmentError, StepScorer, StepScores

MODEL_ID = "UKPLab/Qwen2.5-3b-spare-prm-math"
MODEL_REVISION = "a4f77040cadc6a0f5f7758ec42d3f999376aef4a"
SIGNAL = "prm_correct_prob"

# Verbatim from the model card ("include this instruction as it was left as it
# is during the PRM training").
SPARE_INSTRUCTION = (
    "You are an expert at solving challenging math problems spanning across various categories "
    "and difficulties such as Algebra, Number Theory, Geometry, Counting and Probability, "
    "Precalculus etc. For a given math problem, your task is to generate a step-by-step "
    "reasoning-based solution providing an answer to the question. Identify the correct concepts, "
    "formulas and heuristics that needs to be applied and then derive the contents of the reasoning "
    "steps from the given contexts and accurate calculations from the previous reasoning steps."
)


@dataclass(frozen=True)
class SpareFormat:
    instruction: str = SPARE_INSTRUCTION
    reasoning_trigger: str = "Let's think step by step.\n"
    step_tag: str = " ки"          # leading space required (model card)
    step_separator: str = "\n"     # tag + separator = " ки\n", as in training
    negative_token: str = "-"
    positive_token: str = "+"
    strip_steps: bool = True        # training regex consumed whitespace around step markers

    def step_text(self, step: str) -> str:
        return step.strip() if self.strip_steps else step

    def assistant_content(self, steps: Sequence[str]) -> str:
        body = "".join(f"{self.step_text(s)}{self.step_tag}{self.step_separator}" for s in steps)
        return self.reasoning_trigger + body

    def messages(self, problem: str, steps: Sequence[str]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": self.assistant_content(steps)},
        ]


@dataclass(frozen=True)
class TokenIds:
    step_tag: int
    negative: int
    positive: int


def resolve_token_ids(tokenizer: Any, fmt: SpareFormat) -> TokenIds:
    """Token ids exactly as in the model card."""
    step_tag = tokenizer.encode(fmt.step_tag)[-1]
    negative, positive = tokenizer.convert_tokens_to_ids([fmt.negative_token, fmt.positive_token])
    unk = getattr(tokenizer, "unk_token_id", None)
    for name, tid in (("negative", negative), ("positive", positive)):
        if tid is None or (unk is not None and tid == unk):
            raise ValueError(f"{name} target token is not a single vocabulary token")
    if len({step_tag, negative, positive}) != 3:
        raise ValueError("step tag and target tokens must be distinct")
    return TokenIds(step_tag=step_tag, negative=negative, positive=positive)


def encode_example(tokenizer: Any, fmt: SpareFormat, example: ProcessBenchExample) -> list[int]:
    """Chat-template text, then tokenize without adding special tokens
    (identical ids to apply_chat_template(tokenize=True) for this tokenizer;
    the authors' batch code also tokenizes the rendered text)."""
    text = tokenizer.apply_chat_template(fmt.messages(example.problem, example.steps), tokenize=False)
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def tag_positions(input_ids: Sequence[int], step_tag_id: int) -> list[int]:
    return [i for i, t in enumerate(input_ids) if t == step_tag_id]


def check_alignment(positions: Sequence[int], num_steps: int) -> None:
    if len(positions) != num_steps:
        raise AlignmentError(f"found {len(positions)} step-tag tokens for {num_steps} steps")


def positive_probability(negative_logit: float, positive_logit: float) -> float:
    """softmax([neg, pos])[1], computed stably."""
    d = negative_logit - positive_logit
    if d >= 0:
        e = math.exp(-d)
        return e / (1.0 + e)
    return 1.0 / (1.0 + math.exp(d))


def scores_from_logit_pairs(pairs: Iterable[tuple[float, float]]) -> list[float]:
    """[(logit('-'), logit('+')) per tag] -> [P('+') per step]."""
    return [positive_probability(neg, pos) for neg, pos in pairs]


def batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class SparePRMScorer(StepScorer):
    """Scores ProcessBench steps with SPARE-PRM.

    ``tokenizer`` / ``model`` can be injected (tests); otherwise they are loaded
    from the Hub at ``revision`` on first use. Model weights are only loaded
    when ``score`` needs them.
    """

    name = "spare_prm"
    signal_names = (SIGNAL,)

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: str | None = MODEL_REVISION,
        *,
        dtype: str = "bfloat16",
        device: str = "auto",
        batch_size: int = 1,
        max_tokens: int | None = 4096,
        fmt: SpareFormat | None = None,
        tokenizer: Any = None,
        model: Any = None,
    ):
        if revision is None:
            raise ValueError("SPARE-PRM revision must be pinned")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model_id, self.revision = model_id, revision
        self.dtype, self.device_spec = dtype, device
        self.batch_size, self.max_tokens = batch_size, max_tokens
        self.fmt = fmt or SpareFormat()
        self._tokenizer, self._model = tokenizer, model
        self._token_ids: TokenIds | None = None
        self._device: Any = None

    # ------------------------------------------------------------------ setup

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        return self._tokenizer

    @property
    def token_ids(self) -> TokenIds:
        if self._token_ids is None:
            self._token_ids = resolve_token_ids(self.tokenizer, self.fmt)
        return self._token_ids

    @property
    def model(self) -> Any:
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, revision=self.revision, dtype=_torch_dtype(self.dtype)
            )
            self._model.to(self.device).eval()
            torch.set_grad_enabled(False)
        return self._model

    @property
    def device(self) -> Any:
        if self._device is None:
            self._device = resolve_device(self.device_spec)
        return self._device

    # ---------------------------------------------------------------- scoring

    def prepare(self, example: ProcessBenchExample) -> tuple[list[int], list[int]]:
        """Token ids and the step-tag positions, with alignment and length checks."""
        ids = encode_example(self.tokenizer, self.fmt, example)
        if self.max_tokens is not None and len(ids) > self.max_tokens:
            raise ValueError(f"input has {len(ids)} tokens > max_tokens={self.max_tokens}")
        positions = tag_positions(ids, self.token_ids.step_tag)
        check_alignment(positions, example.num_steps)
        return ids, positions

    def score(self, examples: Sequence[ProcessBenchExample]) -> list[StepScores]:
        results: list[StepScores | None] = [None] * len(examples)
        ready = []
        for i, example in enumerate(examples):
            try:
                ids, positions = self.prepare(example)
            except Exception as exc:  # per-example failure, keep going
                results[i] = StepScores.failure(example, exc)
                continue
            ready.append((i, example, ids, positions))

        for chunk in batched(ready, self.batch_size):
            for i, result in self._score_chunk(chunk):
                results[i] = result
        return [r for r in results if r is not None]

    def _score_chunk(self, chunk: Sequence[tuple]) -> list[tuple[int, StepScores]]:
        try:
            pair_lists = self.logit_pairs([c[2] for c in chunk], [c[3] for c in chunk])
        except Exception as exc:
            if len(chunk) > 1:  # e.g. OOM: retry one at a time so one example can't sink the batch
                _free_cuda_cache()
                return [r for c in chunk for r in self._score_chunk([c])]
            i, example, ids, _ = chunk[0]
            _free_cuda_cache()
            return [(i, StepScores.failure(example, exc, num_tokens=len(ids)))]

        out = []
        for (i, example, ids, _), pairs in zip(chunk, pair_lists):
            result = StepScores(
                id=example.id,
                num_steps=example.num_steps,
                signals={SIGNAL: scores_from_logit_pairs(pairs)},
                num_tokens=len(ids),
            )
            out.append((i, result.check(self.signal_names)))
        return out

    def logit_pairs(
        self, batch_ids: Sequence[Sequence[int]], batch_positions: Sequence[Sequence[int]]
    ) -> list[list[tuple[float, float]]]:
        """Forward pass -> [(logit('-'), logit('+')) at each tag] per sequence.

        Right padding (as in the authors' code) + causal attention means pads
        never influence tag positions. The LM head is applied only at tag
        positions — the same dot products as full logits, far less memory.
        """
        import torch

        model, tids = self.model, self.token_ids
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer has no pad token")
        width = max(len(ids) for ids in batch_ids)
        input_ids = torch.full((len(batch_ids), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(batch_ids), width), dtype=torch.long)
        for b, ids in enumerate(batch_ids):
            input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[b, : len(ids)] = 1

        with torch.inference_mode():
            hidden = model.base_model(
                input_ids=input_ids.to(self.device), attention_mask=attention.to(self.device)
            )[0]
            head = model.get_output_embeddings()
            targets = [tids.negative, tids.positive]
            out = []
            for b, positions in enumerate(batch_positions):
                logits = head(hidden[b, list(positions)])[:, targets].float().cpu().tolist()
                out.append([(neg, pos) for neg, pos in logits])
        return out

    # ------------------------------------------------------------- metadata

    def runtime_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "scorer": self.name,
            "signals": list(self.signal_names),
            "model_id": self.model_id,
            "model_revision": self.revision,
            "dtype": self.dtype,
            "device": str(self._device) if self._device is not None else self.device_spec,
            "batch_size": self.batch_size,
            "max_tokens": self.max_tokens,
            "padding_side": "right",
            "format": asdict(self.fmt),
            "token_ids": asdict(self.token_ids),
            "score_definition": "softmax([logit('-'), logit('+')])[1] at each step-tag token, float32",
        }
        if self._model is not None:
            info["attn_implementation"] = getattr(self._model.config, "_attn_implementation", None)
        return info


def resolve_device(spec: str) -> Any:
    import torch

    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {spec!r} requested but CUDA is not available")
        if (device.index or 0) >= torch.cuda.device_count():
            raise RuntimeError(f"device {spec!r} requested but only {torch.cuda.device_count()} GPU(s) visible")
    return device


def _torch_dtype(name: str) -> Any:
    import torch

    mapping = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in mapping:
        raise ValueError(f"dtype must be one of {sorted(mapping)}, got {name!r}")
    return mapping[name]


def _free_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
