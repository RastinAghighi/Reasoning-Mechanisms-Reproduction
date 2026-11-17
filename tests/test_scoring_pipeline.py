"""Partitioning, resumable score files, held-out guard, and the score_spare.py runner."""

import json
import runpy

import pytest

from fakes import FakeTokenizer
from qmind_reasoning.data import processbench as pb
from qmind_reasoning.paths import CONFIGS_DIR, EXPERIMENTS_DIR
from qmind_reasoning.scoring.base import StepScores
from qmind_reasoning.scoring.io import HeldOutAccessError, ScoreWriter, load_score_records, make_record
from test_spare_prm import FakeLogitScorer, expected_scores


def synthetic_gsm8k():
    """400 records with the real class counts (193 correct, 207 error, 7 error-with-correct-answer)."""
    records = []
    for i in range(400):
        if i < 193:
            label, final = -1, True
        else:
            label, final = (i % 3), i < 200
        steps = [f"Step {k} for {i}." for k in range(3 + i % 4)]
        records.append({"id": f"gsm8k-{i}", "generator": "g", "problem": f"Problem {i}?", "steps": steps,
                        "final_answer_correct": final, "label": label})
    return records


@pytest.fixture
def examples():
    return pb.validate_split(synthetic_gsm8k(), "gsm8k").valid


# --------------------------------------------------------------------------- #
# Dev / held-out partition
# --------------------------------------------------------------------------- #

def test_partition_is_dev_profile_sample_plus_rest(examples):
    part = pb.partition_dev_heldout(examples, 100, seed=42)
    dev = {i for i, p in part.items() if p == pb.DEV}
    held = {i for i, p in part.items() if p == pb.HELDOUT}
    assert len(dev) == 100 and len(held) == 300 and not dev & held
    assert dev | held == {e.id for e in examples}
    assert dev == {e.id for e in pb.sample_examples(examples, 100, 42)}
    assert {e.id for e in pb.sample_examples(examples, 20, 42)} <= dev  # smoke inside dev


def test_partition_rejects_bad_sizes(examples):
    for size in (0, 400):
        with pytest.raises(ValueError):
            pb.partition_dev_heldout(examples, size, seed=1)


# --------------------------------------------------------------------------- #
# Score files
# --------------------------------------------------------------------------- #

def record(i, partition="dev", error=None):
    ex = pb.ProcessBenchExample(f"gsm8k-{i}", "g", "p", ("a", "b"), True, -1, i)
    res = StepScores(ex.id, 2, {} if error else {"prm_correct_prob": [0.9, 0.8]}, 10, error)
    return make_record(res, ex, "gsm8k", partition, "spare_prm", "m", "rev")


def test_writer_append_and_resume(tmp_path):
    w = ScoreWriter(tmp_path / "s.jsonl")
    w.append([record(0), record(1, error="boom")])
    w2 = ScoreWriter(tmp_path / "s.jsonl")  # "restart"
    assert w2.completed_ids() == {"gsm8k-0", "gsm8k-1"}
    assert w2.completed_ids(include_errors=False) == {"gsm8k-0"}
    assert w2.drop_errors() == 1 and w2.completed_ids() == {"gsm8k-0"}


def test_writer_discards_torn_last_line(tmp_path):
    path = tmp_path / "s.jsonl"
    ScoreWriter(path).append([record(0), record(1)])
    path.write_bytes(path.read_bytes()[:-25])  # crash mid-write
    assert ScoreWriter(path).completed_ids() == {"gsm8k-0"}


def test_record_contents_and_id_check():
    r = record(3)
    assert {k: r[k] for k in ("id", "label", "num_steps", "partition", "model_revision")} == {
        "id": "gsm8k-3", "label": -1, "num_steps": 2, "partition": "dev", "model_revision": "rev"}
    ex = pb.ProcessBenchExample("gsm8k-9", "g", "p", ("a",), True, -1, 9)
    with pytest.raises(ValueError, match="does not belong"):
        make_record(StepScores("gsm8k-8", 1), ex, "gsm8k", "dev", "s", "m", "r")


def test_heldout_labels_require_final_evaluation(tmp_path):
    path = tmp_path / "s.jsonl"
    ScoreWriter(path).append([record(0, "dev"), record(1, "heldout"), record(2, "dev", error="x")])
    assert [r["id"] for r in load_score_records(path, "dev")] == ["gsm8k-0"]
    with pytest.raises(HeldOutAccessError):
        load_score_records(path, "heldout")
    assert [r["id"] for r in load_score_records(path, "heldout", final_evaluation=True)] == ["gsm8k-1"]


# --------------------------------------------------------------------------- #
# Runner end to end (synthetic data, fake tokenizer and forward pass)
# --------------------------------------------------------------------------- #

@pytest.fixture
def runner(tmp_path, monkeypatch):
    data = tmp_path / "gsm8k.json"
    data.write_text(json.dumps(synthetic_gsm8k()), encoding="utf-8")
    monkeypatch.setattr(pb, "fetch_split", lambda split, revision, cache_dir=None, timeout=60.0: data)
    module = runpy.run_path(str(EXPERIMENTS_DIR / "01_error_localization" / "score_spare.py"))
    out = tmp_path / "out"
    state = {"poison": set()}

    def factory(config):
        tok = FakeTokenizer()
        poison = {tok._id(w) for w in state["poison"]}
        opts = config.params["spare"]
        return FakeLogitScorer(tokenizer=tok, poison=poison, revision=config.models["prm"].revision,
                               model_id=config.models["prm"].id, batch_size=opts["batch_size"])

    def run(*args):
        return module["main"](list(args), scorer_factory=factory)

    def run_dir():
        (d,) = (out / "01_error_localization").iterdir()
        return d

    return run, run_dir, out, state


def smoke_args(out, *extra):
    return ["--config", str(CONFIGS_DIR / "01_error_localization_smoke.yaml"),
            "--set", f"output_root={out}", *extra]


def read_scores(run_dir):
    return [json.loads(l) for l in (run_dir / "artifacts" / "spare_gsm8k_scores.jsonl").read_text(encoding="utf-8").splitlines()]


def test_runner_smoke_end_to_end(runner):
    run, run_dir, out, _ = runner
    assert run(*smoke_args(out, "--set", "params.spare.batch_size=4")) == 0
    d = run_dir()
    records = read_scores(d)
    manifest = json.loads((d / "artifacts" / "processbench_gsm8k_manifest.json").read_text(encoding="utf-8"))
    partition = json.loads((d / "artifacts" / "processbench_gsm8k_partition.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in records] == manifest["sample_ids"] and len(records) == 20
    assert len(partition["dev_ids"]) == 100 and len(partition["heldout_ids"]) == 300
    assert all(r["partition"] == "dev" for r in records)  # smoke sample lies inside dev
    labels = {e["id"]: e["label"] for e in synthetic_gsm8k()}
    for r in records:
        assert r["label"] == labels[r["id"]] and r["error"] is None
        assert r["signals"]["prm_correct_prob"] == pytest.approx(expected_scores(r["num_steps"]))
        assert r["model_revision"] == "a4f77040cadc6a0f5f7758ec42d3f999376aef4a"
    runtime = json.loads((d / "artifacts" / "spare_gsm8k_runtime.json").read_text(encoding="utf-8"))
    assert runtime[0]["runtime"]["batch_size"] == 4
    assert json.loads((d / "metadata.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_runner_resume_after_interruption(runner):
    run, run_dir, out, _ = runner
    assert run(*smoke_args(out)) == 0
    d = run_dir()
    scores = d / "artifacts" / "spare_gsm8k_scores.jsonl"
    lines = scores.read_text(encoding="utf-8").splitlines(keepends=True)
    scores.write_text("".join(lines[:12]) + lines[12][:30], encoding="utf-8")  # 12 done + torn line
    assert run("--resume", str(d)) == 0
    records = read_scores(d)
    assert len(records) == 20 and len({r["id"] for r in records}) == 20
    runtime = json.loads((d / "artifacts" / "spare_gsm8k_runtime.json").read_text(encoding="utf-8"))
    assert [s["pending"] for s in runtime] == [20, 8]


def test_runner_records_errors_and_retries(runner):
    run, run_dir, out, state = runner
    state["poison"] = {"Problem"}  # every example fails in the forward pass
    assert run(*smoke_args(out)) == 1
    d = run_dir()
    assert all(r["error"] for r in read_scores(d))
    assert json.loads((d / "metadata.json").read_text(encoding="utf-8"))["status"] == "completed_with_errors"
    state["poison"] = set()
    assert run("--resume", str(d), "--retry-errors") == 0
    assert all(r["error"] is None for r in read_scores(d)) and len(read_scores(d)) == 20


def test_runner_tokenize_only_writes_no_scores(runner):
    run, run_dir, out, _ = runner
    assert run(*smoke_args(out, "--tokenize-only")) == 0
    art = run_dir() / "artifacts"
    report = json.loads((art / "spare_gsm8k_tokenization.json").read_text(encoding="utf-8"))
    assert report["num_aligned"] == 20 and not report["failures"]
    assert not (art / "spare_gsm8k_scores.jsonl").exists()


def test_resume_rejects_overrides(runner):
    run, run_dir, out, _ = runner
    run(*smoke_args(out))
    with pytest.raises(SystemExit):
        run("--resume", str(run_dir()), "--set", "seed=1")
