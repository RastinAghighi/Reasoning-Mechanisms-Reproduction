"""SPARE-PRM formatting, tokenization, score extraction, alignment, failures (no downloads)."""

import math

import pytest

from fakes import FakeTokenizer
from qmind_reasoning.data.processbench import ProcessBenchExample
from qmind_reasoning.scoring.base import AlignmentError, StepScores
from qmind_reasoning.scoring.spare_prm import (
    SPARE_INSTRUCTION,
    SparePRMScorer,
    SpareFormat,
    check_alignment,
    encode_example,
    positive_probability,
    resolve_token_ids,
    scores_from_logit_pairs,
    tag_positions,
)

# --- the authors' published example (model card / README), verbatim -------------
CARD_CORRECT = (
    "Let's think step by step.\nYann can order 1 of the 10 dishes. ки\nWhen he picks a dish, there are 9 left "
    "for Camille to choose from. ки\nThus, there are $10\\cdot 9=\\boxed{90}$ possible combinations.\nHence, "
    "the answer is 90. ки\n"
)
CARD_INCORRECT = (
    "Let's think step by step.\nWithout any restrictions, Yann and Camille could both order the same dish out of "
    "the 10 options, for a total of $10 \\cdot 9$ dishes. ки\nHowever, since Yann orders one of the 9 dishes that "
    "Camille didn't order (and vice versa), the number of possible combinations becomes $10 \\cdot 9 - 8 = "
    "\\boxed{72}$.\nHence, the answer is 72. ки\n"
)
CARD_INSTRUCTION_START = "You are an expert at solving challenging math problems spanning across various categories"


def card_steps(generation):
    """Split the card's assistant text back into ProcessBench-style steps."""
    body = generation.removeprefix("Let's think step by step.\n")
    return body.split(" ки\n")[:-1]


def example(i=0, steps=("a = 1.", "b = a + 1 = 2.", "So the answer is 2."), label=-1, problem="What is b?"):
    return ProcessBenchExample(
        id=f"gsm8k-{i}", generator="g", problem=problem, steps=tuple(steps),
        final_answer_correct=label == -1, label=label, index=i,
    )


class FakeLogitScorer(SparePRMScorer):
    """Replaces the forward pass: logit('+') - logit('-') = 0.1 * (step index) - 0.2,
    so every score reveals which step it was computed for."""

    def __init__(self, poison=(), **kwargs):
        kwargs.setdefault("tokenizer", FakeTokenizer())
        super().__init__(**kwargs)
        self.poison, self.calls = set(poison), []

    def logit_pairs(self, batch_ids, batch_positions):
        self.calls.append(len(batch_ids))
        out = []
        for ids, positions in zip(batch_ids, batch_positions):
            if any(tok in self.poison for tok in ids):
                raise RuntimeError("simulated CUDA out of memory")
            out.append([(0.0, 0.1 * k - 0.2) for k in range(len(positions))])
        return out


def expected_scores(n):
    return [positive_probability(0.0, 0.1 * k - 0.2) for k in range(n)]


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("generation", [CARD_CORRECT, CARD_INCORRECT])
def test_format_reproduces_model_card_generation_exactly(generation):
    assert SpareFormat().assistant_content(card_steps(generation)) == generation


def test_messages_structure_and_instruction():
    fmt = SpareFormat()
    msgs = fmt.messages("Q?", ["s0", "s1"])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == SPARE_INSTRUCTION
    assert SPARE_INSTRUCTION.startswith(CARD_INSTRUCTION_START)
    assert SPARE_INSTRUCTION.endswith("accurate calculations from the previous reasoning steps.")
    assert msgs[1]["content"] == "Q?"  # problem verbatim
    assert msgs[2]["content"] == "Let's think step by step.\ns0 ки\ns1 ки\n"


def test_steps_are_stripped_but_internal_text_kept():
    fmt = SpareFormat()
    assert fmt.assistant_content(["  x = 1\ny = 2 \n"]) == "Let's think step by step.\nx = 1\ny = 2 ки\n"
    raw = SpareFormat(strip_steps=False)
    assert raw.assistant_content([" x "]) == "Let's think step by step.\n x  ки\n"


def test_reasoning_trigger_configurable():
    assert SpareFormat(reasoning_trigger="").assistant_content(["a"]) == "a ки\n"


# --------------------------------------------------------------------------- #
# Tokenization and alignment
# --------------------------------------------------------------------------- #

def test_token_ids_follow_model_card_rule():
    tok = FakeTokenizer()
    ids = resolve_token_ids(tok, SpareFormat())
    assert tok.encode(" ки") == [tok.vocab[" к"], tok.vocab["и"]]
    assert ids.step_tag == tok.vocab["и"]  # encode(" ки")[-1]
    assert (ids.negative, ids.positive) == (tok.vocab["-"], tok.vocab["+"])


def test_token_ids_reject_unknown_targets():
    with pytest.raises(ValueError, match="single vocabulary token"):
        resolve_token_ids(FakeTokenizer(), SpareFormat(positive_token="++"))


def test_one_tag_per_step_at_step_end():
    tok, fmt = FakeTokenizer(), SpareFormat()
    ex = example()
    ids = encode_example(tok, fmt, ex)
    tag = resolve_token_ids(tok, fmt).step_tag
    positions = tag_positions(ids, tag)
    check_alignment(positions, ex.num_steps)
    inv = {v: k for k, v in tok.vocab.items()}
    for k, pos in enumerate(positions):
        prefix = "".join(inv[t] for t in ids[: pos + 1])
        assert prefix.endswith(ex.steps[k] + " ки")   # tag k closes step k
        assert inv[ids[pos + 1]] == "\n"


def test_chat_text_then_tokenize_equals_direct_template():
    tok, fmt, ex = FakeTokenizer(), SpareFormat(), example()
    direct = tok.apply_chat_template(fmt.messages(ex.problem, ex.steps), tokenize=True)
    assert encode_example(tok, fmt, ex) == direct


def test_check_alignment_raises_on_mismatch():
    check_alignment([4, 9], 2)
    with pytest.raises(AlignmentError, match="found 3 step-tag tokens for 2 steps"):
        check_alignment([1, 4, 9], 2)


# --------------------------------------------------------------------------- #
# Score extraction
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("neg, pos", [(0.0, 0.0), (1.5, -2.0), (-3.0, 4.0), (1000.0, -1000.0), (-1000.0, 1000.0)])
def test_positive_probability_is_two_way_softmax(neg, pos):
    p = positive_probability(neg, pos)
    m = max(neg, pos)
    expected = math.exp(pos - m) / (math.exp(neg - m) + math.exp(pos - m))
    assert p == pytest.approx(expected, abs=1e-12)
    assert 0.0 <= p <= 1.0


def test_scores_keep_step_order():
    assert scores_from_logit_pairs([(0, 5), (5, 0), (0, 0)]) == pytest.approx(
        [positive_probability(0, 5), positive_probability(5, 0), 0.5]
    )


# --------------------------------------------------------------------------- #
# Scorer behaviour (fake forward pass)
# --------------------------------------------------------------------------- #

def test_scorer_aligns_scores_with_steps_and_input_order():
    exs = [example(i, steps=[f"step {k} of {i}." for k in range(n)]) for i, n in enumerate([3, 1, 5, 2])]
    scorer = FakeLogitScorer(batch_size=3)
    results = scorer.score(exs)
    assert [r.id for r in results] == [e.id for e in exs]
    for r, e in zip(results, exs):
        assert r.ok and r.num_steps == e.num_steps
        assert r.signals["prm_correct_prob"] == pytest.approx(expected_scores(e.num_steps))
        assert r.num_tokens > 0
    assert scorer.calls == [3, 1]  # batching respected


def test_stray_tag_token_is_reported_not_misaligned():
    good = example(0)
    stray = example(1, steps=["Ни один.", "ok."])                # contains the tag token "и"
    in_problem = example(2, problem="Write ки here?")           # tag text in the problem
    results = FakeLogitScorer().score([good, stray, in_problem])
    assert results[0].ok
    for r in results[1:]:
        assert not r.ok and r.error.startswith("AlignmentError") and r.signals == {}


def test_overlong_input_is_reported_not_truncated():
    (r,) = FakeLogitScorer(max_tokens=10).score([example()])
    assert not r.ok and "max_tokens=10" in r.error


def test_batch_failure_falls_back_to_single_examples():
    tok = FakeTokenizer()
    poison_id = tok._id("POISON")
    exs = [example(0), example(1, steps=["POISON step."]), example(2)]
    scorer = FakeLogitScorer(tokenizer=tok, poison={poison_id}, batch_size=3)
    results = scorer.score(exs)
    assert [r.ok for r in results] == [True, False, True]
    assert "simulated CUDA out of memory" in results[1].error
    assert scorer.calls == [3, 1, 1, 1]
    assert results[0].signals["prm_correct_prob"] == pytest.approx(expected_scores(3))


def test_step_scores_check_converts_wrong_lengths_to_errors():
    r = StepScores(id="x", num_steps=3, signals={"prm_correct_prob": [0.1, 0.2]}).check(["prm_correct_prob"])
    assert not r.ok and "wrong lengths" in r.error and r.signals == {}
    r = StepScores(id="x", num_steps=2, signals={}).check(["prm_correct_prob"])
    assert "missing signals" in r.error


def test_scorer_requires_pinned_revision_and_valid_batch():
    with pytest.raises(ValueError, match="pinned"):
        SparePRMScorer(revision=None)
    with pytest.raises(ValueError, match="batch_size"):
        SparePRMScorer(batch_size=0)


def test_runtime_info_records_reproducibility_fields():
    info = FakeLogitScorer(revision="abc", batch_size=2, device="cpu").runtime_info()
    assert info["model_revision"] == "abc" and info["batch_size"] == 2
    assert info["format"]["step_tag"] == " ки" and info["format"]["instruction"] == SPARE_INSTRUCTION
    assert set(info["token_ids"]) == {"step_tag", "negative", "positive"}
