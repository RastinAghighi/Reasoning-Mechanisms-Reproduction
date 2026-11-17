import json
import random

import pytest
import yaml

from qmind_reasoning.config import load_config
from qmind_reasoning.paths import CONFIGS_DIR, EXPERIMENTS_DIR, PROJECT_ROOT
from qmind_reasoning.runs import init_run
from qmind_reasoning.seeding import set_seed


@pytest.fixture
def config():
    return load_config(CONFIGS_DIR / "01_error_localization.yaml", ["seed=3"])


def test_repository_structure_exists():
    for rel in [
        "src/qmind_reasoning",
        "tests",
        "configs",
        "docs",
        "results",
        "experiments/01_error_localization",
        "experiments/02_dcot_refinement",
    ]:
        assert (PROJECT_ROOT / rel).is_dir(), rel
    for rel in ["docs/RESEARCH_PLAN.md", "docs/EXPERIMENT_LOG.md", "README.md", ".gitignore"]:
        assert (PROJECT_ROOT / rel).is_file(), rel
    for exp in ["01_error_localization", "02_dcot_refinement"]:
        assert (EXPERIMENTS_DIR / exp / "run.py").is_file()
        assert (EXPERIMENTS_DIR / exp / "README.md").is_file()


def test_init_run_creates_layout(tmp_path, config):
    run = init_run(config, overrides=["seed=3"], output_root=tmp_path)

    assert run.run_dir.parent == tmp_path / config.name
    assert run.artifacts_dir.is_dir() and run.logs_dir.is_dir()

    saved = yaml.safe_load((run.run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert saved == config.to_dict()

    meta = json.loads(run.metadata_path.read_text(encoding="utf-8"))
    assert meta["run_id"] == run.run_id
    assert meta["seed"] == 3
    assert meta["config_fingerprint"] == config.fingerprint()
    assert meta["overrides"] == ["seed=3"]
    assert meta["status"] == "initialized"
    assert "available" in meta["git"]
    assert "models.prm.id" in meta["unpinned_revisions"]


def test_run_ids_do_not_collide(tmp_path, config):
    first = init_run(config, output_root=tmp_path)
    second = init_run(config, output_root=tmp_path)
    assert first.run_dir != second.run_dir


def test_finish_updates_metadata(tmp_path, config):
    run = init_run(config, output_root=tmp_path)
    run.finish(status="completed", metrics={"f1": 0.5})
    meta = run.read_metadata()
    assert meta["status"] == "completed"
    assert meta["metrics"] == {"f1": 0.5}
    assert "finished_utc" in meta


def test_set_seed_is_reproducible():
    set_seed(123)
    a = [random.random() for _ in range(3)]
    set_seed(123)
    assert a == [random.random() for _ in range(3)]


@pytest.mark.parametrize("experiment", ["01_error_localization", "02_dcot_refinement"])
def test_entry_point_dry_run(experiment, capsys):
    import runpy

    module = runpy.run_path(str(EXPERIMENTS_DIR / experiment / "run.py"))
    assert module["main"](["--dry-run"]) == 0
    assert experiment in capsys.readouterr().out
