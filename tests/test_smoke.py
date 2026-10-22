import importlib
import json
import random

import pytest

from tinyops.utils.envinfo import collect_env
from tinyops.utils.envinfo import main as envinfo_main
from tinyops.utils.io import read_jsonl, sha256_file, write_jsonl
from tinyops.utils.paths import ROOT
from tinyops.utils.seed import set_seed


@pytest.mark.parametrize(
    "mod",
    ["sim", "scenarios", "policies", "scoring", "train", "eval", "report", "utils"],
)
def test_subpackages_import(mod):
    importlib.import_module(f"tinyops.{mod}")


def test_jsonl_roundtrip_unicode_nested(tmp_path):
    rows = [{"a": "héllo ✓ 日本", "n": {"x": [1, 2, {"y": None}]}}, {"b": 2}]
    p = tmp_path / "deep" / "dir" / "x.jsonl"
    write_jsonl(p, rows)
    assert read_jsonl(p) == rows


def test_sha256_stable_and_sensitive(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("abc")
    h1 = sha256_file(p)
    assert sha256_file(p) == h1
    p.write_text("abd")
    assert sha256_file(p) != h1


def test_root_has_pyproject():
    assert (ROOT / "pyproject.toml").exists()


def test_set_seed_reproducible():
    set_seed(1)
    a = random.random()
    set_seed(1)
    assert random.random() == a


def test_envinfo_writes_valid_json(tmp_path):
    out = tmp_path / "env.json"
    envinfo_main(out=out)
    data = json.loads(out.read_text())
    for key in [
        "python", "platform", "packages", "git_commit", "cuda_available", "gpu_name",
        "gpu_mem_gb", "timestamp_utc",
    ]:
        assert key in data
    assert "agenttune" in data["packages"]
    assert collect_env()["python"] == data["python"]
