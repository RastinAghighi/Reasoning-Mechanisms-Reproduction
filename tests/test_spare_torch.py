"""The real forward pass on a tiny, randomly initialised Qwen2 model (no downloads).

Checks that SparePRMScorer.logit_pairs (LM head at tag positions only, right
padding, batching) gives the same scores as the authors' model-card formula:

    logits = model(input_ids).logits[:, :, [neg, pos]]
    scores = logits.softmax(-1)[:, :, 1][input_ids == step_tag_id]
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from fakes import FakeTokenizer  # noqa: E402
from qmind_reasoning.data.processbench import ProcessBenchExample  # noqa: E402
from qmind_reasoning.scoring.spare_prm import SparePRMScorer, encode_example  # noqa: E402


def make_examples():
    texts = [
        ["Tom has 3 apples.", "He buys 2 more, so 3 + 2 = 5.", "The answer is 5."],
        ["x = 4."],
        ["Half of 10 is 5.", "Then 5 * 3 = 15.", "Subtract 1 to get 14.", "So 14.", "Done."],
    ]
    return [
        ProcessBenchExample(f"gsm8k-{i}", "g", f"Question number {i}?", tuple(s), True, -1, i)
        for i, s in enumerate(texts)
    ]


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    tok = FakeTokenizer()
    examples = make_examples()
    probe = SparePRMScorer(tokenizer=tok, model=object(), device="cpu")
    for e in examples:  # grow the fake vocab before sizing the model
        probe.prepare(e)
    config = transformers.Qwen2Config(
        vocab_size=len(tok.vocab) + 8, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512, tie_word_embeddings=True,
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    return tok, model, examples


def card_formula_scores(model, tok, ids, tag_id, neg, pos):
    input_ids = torch.tensor([ids])
    with torch.no_grad():
        logits = model(input_ids).logits[:, :, [neg, pos]]
        scores = logits.float().softmax(dim=-1)[:, :, 1]
        return scores[input_ids == tag_id].tolist()


def test_matches_model_card_formula(setup):
    tok, model, examples = setup
    scorer = SparePRMScorer(tokenizer=tok, model=model, device="cpu", dtype="float32", batch_size=1)
    t = scorer.token_ids
    for e, result in zip(examples, scorer.score(examples)):
        assert result.ok, result.error
        ids = encode_example(tok, scorer.fmt, e)
        expected = card_formula_scores(model, tok, ids, t.step_tag, t.negative, t.positive)
        assert len(expected) == e.num_steps
        assert result.signals["prm_correct_prob"] == pytest.approx(expected, abs=1e-6)


def test_batched_right_padding_equals_unbatched(setup):
    tok, model, examples = setup
    single = SparePRMScorer(tokenizer=tok, model=model, device="cpu", batch_size=1).score(examples)
    batched = SparePRMScorer(tokenizer=tok, model=model, device="cpu", batch_size=3).score(examples)
    for a, b in zip(single, batched):
        assert a.signals["prm_correct_prob"] == pytest.approx(b.signals["prm_correct_prob"], abs=1e-5)


def test_scores_are_deterministic(setup):
    tok, model, examples = setup
    scorer = SparePRMScorer(tokenizer=tok, model=model, device="cpu", batch_size=2)
    assert [r.signals for r in scorer.score(examples)] == [r.signals for r in scorer.score(examples)]
