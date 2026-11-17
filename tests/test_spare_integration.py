"""Opt-in checks against the released SPARE-PRM artifacts.

* QMIND_INTEGRATION=1   real tokenizer (downloads tokenizer files only, ~11 MB)
* QMIND_MODEL_TESTS=1   real weights (~6.2 GB download + 3B forward passes);
                        reproduces the authors' published example scores.
                        Optional QMIND_DEVICE (default "auto").
"""

import os

import pytest

from qmind_reasoning.config import load_config
from qmind_reasoning.data import processbench as pb
from qmind_reasoning.paths import CONFIGS_DIR
from qmind_reasoning.scoring.spare_prm import MODEL_ID, MODEL_REVISION, SparePRMScorer, SpareFormat, encode_example
from test_spare_prm import CARD_CORRECT, CARD_INCORRECT, card_steps

tokenizer_test = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("QMIND_INTEGRATION") != "1", reason="set QMIND_INTEGRATION=1 to run"),
]
model_test = [
    pytest.mark.model,
    pytest.mark.skipif(os.environ.get("QMIND_MODEL_TESTS") != "1", reason="set QMIND_MODEL_TESTS=1 to run"),
]


def apply(marks):
    def wrap(fn):
        for mark in reversed(marks):
            fn = mark(fn)
        return fn
    return wrap


# Verbatim question from the model card (including its "</S>" sentence marker).
CARD_QUESTION = (
    "Yann and Camille go to a restaurant. </S>\nIf there are 10 items on the menu, and each orders one dish, how "
    "many different combinations of meals can Yann and Camille order if they refuse to order the same dish? (It "
    "does matter who orders what---Yann ordering chicken and Camille ordering fish is different from Yann "
    "ordering fish and Camille ordering chicken.)"
)
CARD_EXPECTED = {  # Qwen2.5-3b-spare-prm-math, printed with 4 decimals in the card
    "correct": [0.8710, 0.9163, 0.9786],
    "incorrect": [0.3292, 0.5288],
}
TAG, NEG, POS = 1802, 12, 10  # "и", "-", "+" in the released tokenizer


def card_example(name, generation):
    return pb.ProcessBenchExample(name, "card", CARD_QUESTION, tuple(card_steps(generation)), True, -1, 0)


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("transformers")
    pytest.importorskip("jinja2")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


@apply(tokenizer_test)
def test_real_token_ids(tokenizer):
    assert [tokenizer.decode([t]) for t in tokenizer.encode(" ки")] == [" к", "и"]  # two tokens
    ids = SparePRMScorer(tokenizer=tokenizer).token_ids
    assert (ids.step_tag, ids.negative, ids.positive) == (TAG, NEG, POS)


@apply(tokenizer_test)
def test_card_inputs_match_authors_tokenization(tokenizer):
    fmt = SpareFormat()
    for name, generation in (("correct", CARD_CORRECT), ("incorrect", CARD_INCORRECT)):
        ex = card_example(name, generation)
        authors = tokenizer.apply_chat_template(
            [dict(role="system", content=fmt.instruction), dict(role="user", content=CARD_QUESTION),
             dict(role="assistant", content=generation)],
            tokenize=True,
        )
        authors = list(authors["input_ids"] if isinstance(authors, dict) else authors)
        assert encode_example(tokenizer, fmt, ex) == authors
        assert authors.count(TAG) == ex.num_steps


@apply(tokenizer_test)
def test_all_gsm8k_examples_align(tokenizer):
    cfg = load_config(CONFIGS_DIR / "01_error_localization.yaml")
    records, _ = pb.load_split("gsm8k", cfg.datasets["processbench"].revision)
    examples = pb.validate_split(records, "gsm8k").valid
    scorer = SparePRMScorer(tokenizer=tokenizer)
    lengths = [len(scorer.prepare(e)[0]) for e in examples]  # raises on any misalignment
    assert len(lengths) == 400 and max(lengths) < 2048  # within the PRM's training max_length


@apply(model_test)
@pytest.mark.parametrize("dtype, tol", [("float32", 1e-3), ("bfloat16", 3e-2)])
def test_reproduces_model_card_scores(dtype, tol):
    scorer = SparePRMScorer(dtype=dtype, device=os.environ.get("QMIND_DEVICE", "auto"))
    examples = [card_example("correct", CARD_CORRECT), card_example("incorrect", CARD_INCORRECT)]
    for result in scorer.score(examples):
        assert result.ok, result.error
        assert result.signals["prm_correct_prob"] == pytest.approx(CARD_EXPECTED[result.id], abs=tol)
