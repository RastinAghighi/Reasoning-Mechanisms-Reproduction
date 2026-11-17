"""Unit tests for ProcessBench preparation, using synthetic records only."""

import copy
import json
import runpy

import pytest

from qmind_reasoning.data import processbench as pb
from qmind_reasoning.paths import EXPERIMENTS_DIR


def make_record(i, label=-1, num_steps=4, final_answer_correct=None, split="gsm8k"):
    if final_answer_correct is None:
        final_answer_correct = label == -1
    return {
        "id": f"{split}-{i}",
        "generator": "gen-A" if i % 2 else "gen-B",
        "problem": f"Problem {i}?",
        "steps": [f"step {k} of {i}" for k in range(num_steps)],
        "final_answer_correct": final_answer_correct,
        "label": label,
    }


@pytest.fixture
def records():
    # 30 correct, 50 erroneous with labels cycling through valid indices.
    out = [make_record(i) for i in range(30)]
    out += [make_record(i, label=(i % 4), num_steps=4 + i % 3) for i in range(30, 80)]
    return out


@pytest.fixture
def examples(records):
    return pb.validate_split(records, "synthetic").valid


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def test_read_records_accepts_json_array_and_jsonl(tmp_path, records):
    array = tmp_path / "a.json"
    array.write_text(json.dumps(records), encoding="utf-8")
    lines = tmp_path / "b.jsonl"
    lines.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    assert pb.read_records(array) == pb.read_records(lines) == records


def test_load_split_local_records_provenance(tmp_path, records):
    path = tmp_path / "gsm8k.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    loaded, source = pb.load_split("gsm8k", "rev123", path=path)
    assert loaded == records
    assert source.revision == "rev123"
    assert len(source.sha256) == 64 and source.num_bytes == path.stat().st_size


def test_fetch_requires_pinned_revision_and_known_split():
    with pytest.raises(ValueError, match="revision"):
        pb.fetch_split("gsm8k", "")
    with pytest.raises(ValueError, match="unknown"):
        pb.fetch_split("train", "abc")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_valid_records_pass_unchanged(records):
    original = copy.deepcopy(records)
    report = pb.validate_split(records, "gsm8k")
    assert len(report.valid) == len(records) and not report.invalid
    assert [e.to_record() for e in report.valid] == original
    assert records == original  # input not mutated
    assert [e.index for e in report.valid] == list(range(len(records)))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r.pop("label"), "missing field 'label'"),
        (lambda r: r.update(label=True), "field 'label' has type bool"),
        (lambda r: r.update(label="1"), "field 'label' has type str"),
        (lambda r: r.update(final_answer_correct=1), "final_answer_correct"),
        (lambda r: r.update(steps="one step"), "field 'steps' has type str"),
        (lambda r: r.update(steps=[]), "'steps' is empty"),
        (lambda r: r.update(steps=["a", 2]), "non-string steps"),
        (lambda r: r.update(label=4), "outside valid range"),   # == len(steps): off by one
        (lambda r: r.update(label=-2), "outside valid range"),
        (lambda r: r.update(id=" "), "'id' is empty"),
    ],
)
def test_invalid_records_are_excluded_and_reported(mutate, message):
    good = make_record(0)
    bad = make_record(1)
    mutate(bad)
    report = pb.validate_split([good, bad], "gsm8k")
    assert [e.id for e in report.valid] == ["gsm8k-0"]
    assert len(report.invalid) == 1 and report.invalid[0]["index"] == 1
    assert any(message in err for err in report.invalid[0]["errors"])
    assert not report.ok


def test_non_object_record_rejected():
    report = pb.validate_split(["not a dict"], "gsm8k")
    assert report.invalid[0]["errors"] == ["record is str, not an object"]


def test_label_boundaries_are_valid():
    recs = [make_record(0, label=0), make_record(1, label=3), make_record(2, label=-1)]
    report = pb.validate_split(recs, "gsm8k")
    assert len(report.valid) == 3


def test_duplicate_ids_keep_first_occurrence():
    report = pb.validate_split([make_record(0), make_record(0, label=1)], "gsm8k")
    assert [e.label for e in report.valid] == [-1]
    assert "duplicate id" in report.invalid[0]["errors"][0]


def test_warnings_flag_but_keep_records():
    recs = [
        make_record(0, split="math"),                       # id prefix mismatch
        make_record(1, label=-1, final_answer_correct=False),
        {**make_record(2), "steps": ["ok", "  "]},
    ]
    report = pb.validate_split(recs, "gsm8k")
    assert len(report.valid) == 3 and not report.invalid
    text = json.dumps(report.warnings)
    assert "does not start with 'gsm8k-'" in text
    assert "final_answer_correct is false" in text
    assert "blank steps at indices [1]" in text


def test_expected_split_size_is_checked(records):
    report = pb.validate_split(records, "gsm8k")
    assert report.dataset_issues == ["expected 400 records for 'gsm8k', found 80"]
    assert pb.validate_split(records, "custom").dataset_issues == []


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def test_statistics(examples):
    stats = pb.compute_statistics(examples)
    assert (stats["num_examples"], stats["num_correct"], stats["num_error"]) == (80, 30, 50)
    assert stats["label_vs_final_answer"] == {
        "correct/final_correct": 30, "correct/final_wrong": 0,
        "error/final_correct": 0, "error/final_wrong": 50,
    }
    assert sum(stats["first_error"]["histogram"].values()) == 50
    assert set(stats["first_error"]["histogram"]) == {0, 1, 2, 3}
    assert stats["steps"]["all"]["min"] == 4 and stats["steps"]["all"]["max"] == 6
    assert 0 <= stats["first_error"]["relative_position"]["max"] < 1


def test_statistics_empty():
    stats = pb.compute_statistics([])
    assert stats["num_examples"] == 0 and stats["steps"]["all"] is None


def test_compare_to_paper():
    stats = {"num_error": 207, "num_correct": 193, "label_vs_final_answer": {"error/final_correct": 7}}
    assert pb.compare_to_paper(stats, "gsm8k")["match"] is True
    stats["num_error"] = 206
    result = pb.compare_to_paper(stats, "gsm8k")
    assert result["match"] is False
    assert result["mismatches"] == [{"field": "num_error", "paper": 207, "observed": 206}]
    assert pb.compare_to_paper(stats, "custom")["available"] is False


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def test_sampling_is_deterministic_and_in_file_order(examples):
    a = pb.sample_examples(examples, 20, seed=42)
    b = pb.sample_examples(list(reversed(examples)), 20, seed=42)  # input order irrelevant
    assert [e.id for e in a] == [e.id for e in b]
    assert [e.index for e in a] == sorted(e.index for e in a)
    assert len({e.id for e in a}) == 20


def test_sampling_depends_on_seed(examples):
    ids = lambda s: {e.id for e in pb.sample_examples(examples, 20, seed=s)}
    assert ids(1) != ids(2)


def test_smaller_samples_nest_in_larger(examples):
    smoke = {e.id for e in pb.sample_examples(examples, 20, seed=7)}
    dev = {e.id for e in pb.sample_examples(examples, 50, seed=7)}
    full = {e.id for e in pb.sample_examples(examples, None, seed=7)}
    assert smoke <= dev <= full and len(full) == 80


def test_stratified_sampling_preserves_class_ratio(examples):
    sample = pb.sample_examples(examples, 20, seed=0)  # 30/80 correct -> 7.5 -> 7 or 8
    n_correct = sum(not e.has_error for e in sample)
    assert n_correct in (7, 8) and len(sample) == 20


def test_unstratified_sampling_size(examples):
    assert len(pb.sample_examples(examples, 20, seed=0, stratify=False)) == 20


def test_sampling_full_and_oversized_return_everything(examples):
    assert len(pb.sample_examples(examples, None, seed=0)) == 80
    assert len(pb.sample_examples(examples, 1000, seed=0)) == 80
    with pytest.raises(ValueError):
        pb.sample_examples(examples, 0, seed=0)


def test_sampling_does_not_change_annotations(records, examples):
    by_id = {r["id"]: r for r in records}
    for e in pb.sample_examples(examples, 20, seed=3):
        assert e.to_record() == by_id[e.id]


@pytest.mark.parametrize("sizes, n", [({"a": 3, "b": 7}, 5), ({"a": 193, "b": 207}, 20), ({"a": 1}, 1)])
def test_allocate_sums_to_n(sizes, n):
    alloc = pb.allocate(sizes, n)
    assert sum(alloc.values()) == n
    assert all(0 <= alloc[k] <= sizes[k] for k in sizes)


# --------------------------------------------------------------------------- #
# Manifest and entry point
# --------------------------------------------------------------------------- #

def test_manifest_contents(tmp_path, records):
    path = tmp_path / "gsm8k.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    manifest, sample = pb.prepare_split("gsm8k", "rev", seed=5, sample_size=20, path=path)
    assert manifest["dataset"]["revision"] == "rev"
    assert manifest["sampling"]["seed"] == 5 and manifest["sampling"]["actual_size"] == 20
    assert manifest["sample_ids"] == [e.id for e in sample]
    assert manifest["validation"]["num_valid"] == 80
    assert manifest["statistics"]["sample"]["num_examples"] == 20
    assert manifest["paper_reference_check"]["match"] is False  # synthetic != paper
    assert not pb.manifest_passed(manifest)
    json.dumps(manifest)  # serializable
    assert "ProcessBench `gsm8k`" in pb.format_report(manifest)


def test_prepare_data_script_end_to_end(tmp_path, records):
    data = tmp_path / "custom.json"
    data.write_text(json.dumps([make_record(i, split="custom") for i in range(10)]
                               + [make_record(i, label=1, split="custom") for i in range(10, 25)]),
                    encoding="utf-8")
    script = runpy.run_path(str(EXPERIMENTS_DIR / "01_error_localization" / "prepare_data.py"))
    code = script["main"]([
        "--data-file", str(data),
        "--set", "datasets.processbench.splits=[custom]",
        "--set", "datasets.processbench.sample_size=10",
        "--set", f"output_root={tmp_path / 'out'}",
    ])
    assert code == 0  # custom split: no size/paper reference, all records valid

    (run_dir,) = (tmp_path / "out" / "01_error_localization").iterdir()
    artifacts = run_dir / "artifacts"
    manifest = json.loads((artifacts / "processbench_custom_manifest.json").read_text(encoding="utf-8"))
    sample_lines = (artifacts / "processbench_custom_sample.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(sample_lines) == 10 == len(manifest["sample_ids"])
    assert [json.loads(line)["id"] for line in sample_lines] == manifest["sample_ids"]
    assert (artifacts / "processbench_custom_report.md").is_file()
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed" and meta["data_preparation"]["custom"]["passed"]
