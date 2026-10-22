import json

import pytest

from tinyops.eval.run_eval import POLICIES, run_policy
from tinyops.eval.score_episodes import markdown_table, score_file
from tinyops.scenarios.build_dataset import load_scenarios
from tinyops.utils.io import read_jsonl


def test_run_policy_is_resumable(built_data, tmp_path):
    scs = load_scenarios("val", built_data)[:6]
    out = tmp_path / "ep.jsonl"
    assert run_policy(scs[:3], POLICIES["oracle"], out) == 3
    assert run_policy(scs, POLICIES["oracle"], out) == 3  # only the 3 new ones
    assert run_policy(scs, POLICIES["oracle"], out) == 0
    ids = [r["scenario_id"] for r in read_jsonl(out)]
    assert ids == [s.id for s in scs]


def test_run_policy_is_deterministic_per_seed(built_data, tmp_path):
    scs = load_scenarios("val", built_data)[:5]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_policy(scs, POLICIES["random"], a, seed=5)
    run_policy(scs, POLICIES["random"], b, seed=5)
    assert read_jsonl(a) == read_jsonl(b)


def test_score_file_writes_json_md_and_scores(built_data, tmp_path):
    scs = load_scenarios("val", built_data)
    ep, out = tmp_path / "oracle.jsonl", tmp_path / "m" / "oracle_val.json"
    run_policy(scs, POLICIES["oracle"], ep)
    result = score_file(ep, "val", out, built_data)
    assert result["complete"] and result["n_episodes"] == 180 and result["policies"] == ["oracle"]
    assert result["headline"]["diagnosis_success"]["mean"] == 1.0
    assert set(result["groups"]) == {"failure_type", "difficulty", "injected", "injection_family"}
    saved = json.loads(out.read_text())
    assert saved["headline"]["unsafe_rate"]["mean"] == 0.0
    md = out.with_suffix(".md").read_text()
    assert "Diagnosis success | 100.0%" in md and md.startswith("### oracle_val on val (n=180)")
    assert len(read_jsonl(out.with_suffix(".scores.jsonl"))) == 180
    assert "n/a" in markdown_table("x", result)  # tokens are unavailable for scripted policies


def test_score_file_flags_partial_runs_and_unknown_ids(built_data, tmp_path):
    scs = load_scenarios("val", built_data)
    ep = tmp_path / "e.jsonl"
    run_policy(scs[:10], POLICIES["oracle"], ep)
    assert score_file(ep, "val", tmp_path / "o.json", built_data)["complete"] is False
    with pytest.raises(ValueError, match="not in split"):
        score_file(ep, "test", tmp_path / "o2.json", built_data)
