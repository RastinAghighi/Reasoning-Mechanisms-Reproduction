import pytest

from qmind_reasoning.config import ConfigError, load_config
from qmind_reasoning.paths import CONFIGS_DIR

EXPERIMENT_CONFIGS = ["01_error_localization.yaml", "02_dcot_refinement.yaml"]


@pytest.mark.parametrize("filename", EXPERIMENT_CONFIGS)
def test_shipped_configs_load(filename):
    cfg = load_config(CONFIGS_DIR / filename)
    assert cfg.name == filename.removesuffix(".yaml")
    assert cfg.seed == 42  # inherited from base.yaml
    assert cfg.output_root == "results"
    assert cfg.models and cfg.datasets


def test_experiment_1_contents():
    cfg = load_config(CONFIGS_DIR / "01_error_localization.yaml")
    assert cfg.models["policy"].id == "Qwen/Qwen2.5-Math-7B-Instruct"
    assert cfg.datasets["processbench"].id == "Qwen/ProcessBench"
    assert cfg.datasets["processbench"].splits == ("gsm8k",)
    assert cfg.datasets["processbench"].revision  # pinned
    assert cfg.datasets["processbench"].sample_size is None


@pytest.mark.parametrize("profile, size", [("smoke", 20), ("dev", 100)])
def test_experiment_1_profiles(profile, size):
    cfg = load_config(CONFIGS_DIR / f"01_error_localization_{profile}.yaml")
    full = load_config(CONFIGS_DIR / "01_error_localization.yaml")
    assert cfg.name == full.name and cfg.seed == full.seed
    assert cfg.datasets["processbench"].sample_size == size
    assert cfg.datasets["processbench"].revision == full.datasets["processbench"].revision


def test_experiment_2_contents():
    cfg = load_config(CONFIGS_DIR / "02_dcot_refinement.yaml")
    assert cfg.models["dcot"].id == "microsoft/phi-1_5"
    assert "adapter_id" in cfg.to_dict()["models"]["dcot"]


def test_inheritance_deep_merges(write_yaml):
    write_yaml("parent.yaml", "name: p\nseed: 1\nparams: {a: 1, b: 2}\n")
    child = write_yaml("child.yaml", "extends: parent.yaml\nname: c\nparams: {b: 3}\n")
    cfg = load_config(child)
    assert (cfg.name, cfg.seed, cfg.params) == ("c", 1, {"a": 1, "b": 3})


def test_overrides_are_typed():
    cfg = load_config(
        CONFIGS_DIR / "01_error_localization.yaml",
        ["seed=7", "datasets.processbench.sample_size=20", "tags=[smoke]"],
    )
    assert cfg.seed == 7
    assert cfg.datasets["processbench"].sample_size == 20
    assert cfg.tags == ("smoke",)


def test_fingerprint_is_stable_and_sensitive():
    path = CONFIGS_DIR / "01_error_localization.yaml"
    assert load_config(path).fingerprint() == load_config(path).fingerprint()
    assert load_config(path).fingerprint() != load_config(path, ["seed=1"]).fingerprint()


def test_unpinned_revisions_are_reported(write_yaml):
    path = write_yaml(
        "c.yaml",
        "name: x\nmodels:\n  m: {id: org/model, revision: abc123}\n  n: {id: null}\n"
        "datasets:\n  d: {id: org/data}\n",
    )
    assert load_config(path).unpinned() == ["models.n.id", "datasets.d.revision"]


@pytest.mark.parametrize(
    "text, message",
    [
        ("name: x\nsede: 1\n", "unknown key"),
        ("name: x\nmodels:\n  m: {id: a, revison: b}\n", "unknown key"),
        ("seed: 1\n", "'name' is required"),
        ("name: x\nseed: one\n", "'seed' must be an int"),
        ("name: x\ndatasets:\n  d: {id: a, sample_size: 0}\n", "sample_size"),
        ("- a\n- b\n", "must be a mapping"),
    ],
)
def test_invalid_configs_rejected(write_yaml, text, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write_yaml("bad.yaml", text))


def test_circular_extends_rejected(write_yaml):
    a = write_yaml("a.yaml", "extends: b.yaml\nname: a\n")
    write_yaml("b.yaml", "extends: a.yaml\nname: b\n")
    with pytest.raises(ConfigError, match="circular"):
        load_config(a)


def test_malformed_override_rejected():
    with pytest.raises(ConfigError, match="key=value"):
        load_config(CONFIGS_DIR / "01_error_localization.yaml", ["seed"])
