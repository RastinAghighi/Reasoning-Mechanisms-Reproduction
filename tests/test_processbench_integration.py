"""Optional integration test against the real ProcessBench gsm8k split.

Downloads one ~575 KB JSON file (cached under .cache/datasets). Skipped unless
QMIND_INTEGRATION=1 is set:

    QMIND_INTEGRATION=1 pytest -m integration
"""

import os

import pytest

from qmind_reasoning.config import load_config
from qmind_reasoning.data import processbench as pb
from qmind_reasoning.paths import CONFIGS_DIR

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("QMIND_INTEGRATION") != "1", reason="set QMIND_INTEGRATION=1 to run"),
]

# Immutable file at the pinned revision; a change means the source changed.
GSM8K_SHA256 = "fb1c59dfd1e83e1c5a48dbe598db27e4bc0b8bc524a09445f2d4b69178f61349"


@pytest.fixture(scope="module")
def prepared():
    revision = load_config(CONFIGS_DIR / "01_error_localization.yaml").datasets["processbench"].revision
    return pb.prepare_split("gsm8k", revision, seed=42, sample_size=None)


def test_gsm8k_source_is_pinned(prepared):
    manifest, _ = prepared
    assert manifest["dataset"]["sha256"] == GSM8K_SHA256


def test_gsm8k_validates_and_matches_paper(prepared):
    manifest, sample = prepared
    assert manifest["validation"]["num_records"] == 400
    assert manifest["validation"]["ok"], manifest["validation"]
    assert manifest["paper_reference_check"]["match"], manifest["paper_reference_check"]
    assert len(sample) == 400


def test_gsm8k_documented_example(prepared):
    # Dataset card example: gsm8k-0, Qwen2-7B-Instruct, final answer wrong, label 1.
    _, sample = prepared
    first = sample[0]
    assert (first.id, first.generator, first.final_answer_correct, first.label) == (
        "gsm8k-0", "Qwen2-7B-Instruct", False, 1
    )
    assert first.problem.startswith("Sue lives in a fun neighborhood")


@pytest.mark.parametrize("profile, size", [("smoke", 20), ("dev", 100)])
def test_profiles_sample_nested_subsets(prepared, profile, size):
    _, full = prepared
    cfg = load_config(CONFIGS_DIR / f"01_error_localization_{profile}.yaml")
    assert cfg.datasets["processbench"].sample_size == size
    sample = pb.sample_examples(full, size, cfg.seed)
    assert len(sample) == size
    if profile == "smoke":
        dev = {e.id for e in pb.sample_examples(full, 100, cfg.seed)}
        assert {e.id for e in sample} <= dev
