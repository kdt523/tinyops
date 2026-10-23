import json
import shutil

import pytest

from tests.test_run_eval import StubPolicy
from tinyops.eval import final_eval as fe
from tinyops.eval import run_eval as re_mod
from tinyops.eval.final_eval import can_reuse, run_final
from tinyops.eval.run_eval import run_eval
from tinyops.scenarios.build_dataset import load_scenarios
from tinyops.utils.io import read_json, read_jsonl, sha256_file


def cfg(**policies):
    base = {"seed": 1234, "max_steps": 8, "splits": ["test_ood"], "model": "m/base", "batch_size": 8,
            "max_new_tokens": 256, "policies": policies or {
                "base_zeroshot": {"kind": "hf", "fewshot": 0},
                "final": {"kind": "hf", "adapter": "me/sft+me/final"}}}
    return base


@pytest.fixture
def stub(built_data, monkeypatch):
    scs = load_scenarios("test_ood", built_data)
    uniq = list({s.alert_text: s for s in scs}.values())
    pol = StubPolicy(uniq)
    monkeypatch.setattr(re_mod, "build_policy", lambda *a, **k: pol)
    return pol


def test_final_run_writes_files_and_locks(built_data, tmp_path, stub):
    res = tmp_path / "final"
    rec = run_final(cfg(), data=built_data, results=res, baselines=tmp_path / "none", resolve_revisions=False)
    for pol in ("base_zeroshot", "final"):
        assert (res / "episodes" / f"{pol}_test_ood.jsonl").exists()
        assert (res / "metrics" / f"{pol}_test_ood.json").exists()
    lock = read_json(res / "FINAL_RUN.json")
    assert rec["missing"] == [] and lock["prompt_version"] == "v1" and lock["seed"] == 1234
    assert lock["test_hashes"].count("sha256") == 0 and "test_ood" in lock["test_hashes"]
    assert lock["episodes_sha256"]["episodes/final_test_ood.jsonl"] == sha256_file(res / "episodes" / "final_test_ood.jsonl")
    assert lock["adapter_revisions"]["final"] == {}
    assert "git_commit" in lock and lock["rerun_reason"] is None


def test_second_run_is_refused_without_a_reason(built_data, tmp_path, stub):
    res = tmp_path / "final"
    run_final(cfg(), data=built_data, results=res, baselines=tmp_path / "x", resolve_revisions=False)
    with pytest.raises(FileExistsError, match="locked"):
        run_final(cfg(), data=built_data, results=res, baselines=tmp_path / "x", resolve_revisions=False)
    rec = run_final(cfg(), data=built_data, results=res, baselines=tmp_path / "x",
                    rerun_reason="fixed scorer bug #12", resolve_revisions=False)
    assert read_json(res / "FINAL_RUN.json")["rerun_reason"] == "fixed scorer bug #12" and rec["missing"] == []


def test_tampered_test_data_aborts_before_anything_runs(built_data, tmp_path, stub):
    data = tmp_path / "d"
    shutil.copytree(built_data, data)
    with open(data / "test_ood" / "scenarios.jsonl", "a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="do not match"):
        run_final(cfg(), data=data, results=tmp_path / "final", resolve_revisions=False)
    assert not (tmp_path / "final").exists()


def test_placeholders_and_unknown_kinds_are_rejected(built_data, tmp_path, stub):
    with pytest.raises(ValueError, match="<user>"):
        run_final(cfg(sft={"kind": "hf", "adapter": "<user>/tinyops-tiny-aya-sft"}), data=built_data,
                  results=tmp_path / "f")
    with pytest.raises(ValueError, match="kind"):
        run_final(cfg(x={"kind": "magic"}), data=built_data, results=tmp_path / "f")


def test_partial_sessions_stay_re_runnable_until_complete(built_data, tmp_path, stub):
    res = tmp_path / "final"
    r1 = run_final(cfg(), only=["base_zeroshot"], data=built_data, results=res, resolve_revisions=False)
    assert r1["missing"] == ["final_test_ood"]
    assert (res / "FINAL_RUN.partial.json").exists() and not (res / "FINAL_RUN.json").exists()
    r2 = run_final(cfg(), only=["final"], data=built_data, results=res, resolve_revisions=False)
    assert r2["missing"] == [] and (res / "FINAL_RUN.json").exists()
    assert len(read_jsonl(res / "episodes" / "base_zeroshot_test_ood.jsonl")) == 63  # not re-run


def baseline(built_data, tmp_path, stub, **kw):
    base = tmp_path / "base"
    run_eval("hf", "test_ood", base / "episodes" / "tinyaya_zeroshot_test_ood.jsonl", model="m",
             data=built_data, results=base, **kw)
    return base


def test_baseline_episodes_are_reused_when_compatible(built_data, tmp_path, stub, monkeypatch):
    base = baseline(built_data, tmp_path, stub)
    (base / "episodes").mkdir(exist_ok=True)
    monkeypatch.setattr(re_mod, "build_policy", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must reuse")))
    res = tmp_path / "final"
    rec = run_final(cfg(base_zeroshot={"kind": "hf", "reuse_from": "tinyaya_zeroshot_{split}"}),
                    data=built_data, results=res, baselines=base, resolve_revisions=False)
    assert rec["runs"]["base_zeroshot_test_ood"] == {"reused_from": "tinyaya_zeroshot_test_ood"}
    assert len(read_jsonl(res / "episodes" / "base_zeroshot_test_ood.jsonl")) == 63
    assert (res / "metrics" / "base_zeroshot_test_ood.scores.jsonl").exists()


def test_reuse_conditions(built_data, tmp_path, stub):
    base = baseline(built_data, tmp_path, stub)
    ep, man = base / "episodes" / "tinyaya_zeroshot_test_ood.jsonl", base / "runs" / "tinyaya_zeroshot_test_ood.json"
    c = cfg()
    assert can_reuse(ep, man, "test_ood", c, built_data) is None
    assert "not found" in can_reuse(ep, base / "nope.json", "test_ood", c, built_data)
    assert "seed" in can_reuse(ep, man, "test_ood", {**c, "seed": 7}, built_data)
    assert "different split" in can_reuse(ep, man, "test", c, built_data) or \
        "cover" in can_reuse(ep, man, "test", c, built_data)
    m = json.loads(man.read_text())
    man.write_text(json.dumps({**m, "prompt_version": "v0"}))
    assert "prompt version" in can_reuse(ep, man, "test_ood", c, built_data)
    man.write_text(json.dumps({**m, "limit": 10}))
    assert "partial" in can_reuse(ep, man, "test_ood", c, built_data)
    man.write_text(json.dumps(m))
    ep.write_text("\n".join(ep.read_text().splitlines()[:-3]) + "\n")
    assert "whole split" in can_reuse(ep, man, "test_ood", c, built_data)


def test_committed_final_config_is_complete_and_uses_placeholders():
    import yaml

    with open("configs/final.yaml", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    assert conf["seed"] == 1234 and conf["max_steps"] == 8 and conf["splits"] == ["test", "test_injection", "test_ood"]
    assert set(conf["policies"]) == {"base_zeroshot", "base_fewshot", "sft", "final", "gemini"}
    assert conf["policies"]["gemini"]["splits"] == ["test", "test_injection"]
    with pytest.raises(ValueError, match="<user>"):
        fe.check_config(conf)  # the human must fill the adapter ids before the real run
