import csv
import json
import random

import pytest
import typer

from tinyops.eval import run_eval as re_mod
from tinyops.eval.run_eval import append_compute_log, run_batched, run_eval
from tinyops.policies.base import PolicyOutput, format_tool_call_text
from tinyops.policies.scripted import OraclePolicy
from tinyops.scenarios.build_dataset import load_scenarios
from tinyops.utils.io import read_jsonl


class StubPolicy:
    """Canned texts: replays the oracle's calls, found by the alert text in the conversation."""

    name = "stub"

    def __init__(self, scenarios):
        self.plans = {}
        for sc in scenarios:
            assert sc.alert_text not in self.plans
            self.plans[sc.alert_text] = OraclePolicy(sc.state, random.Random(1)).plan
        self.batch_sizes = []

    def config(self):
        return {"policy": "stub", "name": self.name}

    def act(self, messages):
        plan = self.plans[messages[1]["content"]]
        turn = sum(m["role"] == "assistant" for m in messages)
        call = plan[turn]
        return PolicyOutput(text=format_tool_call_text(call), tool_call=call,
                            usage={"prompt_tokens": 100, "completion_tokens": 10, "latency_s": 0.1})

    def act_batch(self, batch):
        self.batch_sizes.append(len(batch))
        return [self.act(m) for m in batch]


@pytest.fixture
def val5(built_data):
    return load_scenarios("val", built_data)[:5]


def test_batched_run_scores_and_drops_finished_episodes(val5, tmp_path):
    pol = StubPolicy(val5)
    out = tmp_path / "e.jsonl"
    assert run_batched(val5, pol, out, batch_size=2) == 5
    eps = read_jsonl(out)
    assert sorted(e["scenario_id"] for e in eps) == sorted(s.id for s in val5)
    assert all(e["termination"] == "submitted" and e["policy_name"] == "stub" for e in eps)
    assert eps[0]["steps"][0]["prompt_tokens"] == 100
    total_turns = sum(len(e["steps"]) for e in eps)
    assert sum(pol.batch_sizes) == total_turns  # finished episodes leave the batch
    assert pol.batch_sizes[0] == 2 and min(pol.batch_sizes) == 1


def test_resume_skips_finished_scenarios(val5, tmp_path):
    out = tmp_path / "e.jsonl"
    assert run_batched(val5[:3], StubPolicy(val5), out, batch_size=2) == 3
    pol = StubPolicy(val5)
    assert run_batched(val5, pol, out, batch_size=2) == 2
    assert run_batched(val5, pol, out, batch_size=2) == 0
    ids = [e["scenario_id"] for e in read_jsonl(out)]
    assert len(ids) == 5 and len(set(ids)) == 5


def test_failed_generation_is_recorded_not_fatal(val5, tmp_path):
    class Boom(StubPolicy):
        def act_batch(self, batch):
            raise RuntimeError("CUDA out of memory")

    out = tmp_path / "e.jsonl"
    assert run_batched(val5[:2], Boom(val5), out, batch_size=2) == 2
    eps = read_jsonl(out)
    assert all(e["termination"] == "policy_error" and "CUDA out of memory" in e["meta"]["policy_error"]
               for e in eps)


def test_non_batched_policy_failures_are_per_episode(val5, tmp_path):
    class Flaky:
        name = "flaky"

        def __init__(self):
            self.n = 0

        def act(self, messages):
            self.n += 1
            if messages[1]["content"] == val5[0].alert_text:
                raise ValueError("bad")
            return PolicyOutput(text="no idea")

    out = tmp_path / "e.jsonl"
    run_batched(val5[:3], Flaky(), out, batch_size=3)
    terms = {e["scenario_id"]: e["termination"] for e in read_jsonl(out)}
    assert terms[val5[0].id] == "policy_error"
    assert set(terms.values()) == {"policy_error", "parse_failures"}


def test_run_eval_writes_manifest_metrics_and_scores(built_data, tmp_path):
    out, res = tmp_path / "ep" / "oracle_val.jsonl", tmp_path / "results"
    m = run_eval("oracle", "val", out, limit=8, data=built_data, results=res)
    assert m["new_episodes_this_invocation"] == 8 and m["prompt_version"] == "v1"
    for key in ("git_commit", "env", "start_utc", "end_utc", "seed", "split", "policy_config"):
        assert key in m
    manifest = json.loads((res / "runs" / "oracle_val.json").read_text())
    assert manifest["run"] == "oracle_val" and manifest["split"] == "val"
    metrics = json.loads((res / "metrics" / "oracle_val.json").read_text())
    assert metrics["n_episodes"] == 8 and metrics["complete"] is False
    assert (res / "metrics" / "oracle_val.md").exists()
    # resuming does not duplicate and still writes a fresh manifest
    m2 = run_eval("oracle", "val", out, limit=8, data=built_data, results=res)
    assert m2["new_episodes_this_invocation"] == 0
    assert len(read_jsonl(out)) == 8


def test_run_eval_with_a_model_policy_stub(built_data, tmp_path, monkeypatch):
    scs = load_scenarios("val", built_data)[:4]
    monkeypatch.setattr(re_mod, "build_policy", lambda *a, **k: StubPolicy(scs))
    res = tmp_path / "r"
    m = run_eval("hf", "val", tmp_path / "stub_val.jsonl", model="x/y", limit=4, batch_size=2,
                 data=built_data, results=res)
    assert m["policy_config"] == {"policy": "stub", "name": "stub"}
    metrics = json.loads((res / "metrics" / "stub_val.json").read_text())
    assert metrics["headline"]["diagnosis_success"]["mean"] == 1.0
    assert metrics["headline"]["tokens_per_ep"]["n"] == 4  # token usage flows into the metrics


def test_test_split_rules(built_data, tmp_path):
    with pytest.raises(typer.BadParameter, match="--final"):
        run_eval("hf", "test", tmp_path / "x.jsonl", model="m", adapter="some/adapter",
                 data=built_data, results=tmp_path)
    m = run_eval("oracle", "test", tmp_path / "t.jsonl", limit=2, data=built_data, results=tmp_path)
    assert m["split"] == "test"  # scripted/base policies may run there (with a loud warning)


def test_build_policy_validates_inputs():
    with pytest.raises(typer.BadParameter):
        re_mod.build_policy("hf")
    with pytest.raises(typer.BadParameter):
        re_mod.build_policy("api")
    with pytest.raises(typer.BadParameter):
        re_mod.build_policy("nope")


def test_compute_log_header_once(tmp_path):
    append_compute_log(tmp_path, "r1", "T4", 0.5)
    append_compute_log(tmp_path, "r2", "T4", 1.25)
    rows = list(csv.reader((tmp_path / "compute_log.csv").open()))
    assert rows[0] == ["date", "run", "gpu", "hours"]
    assert [r[1:] for r in rows[1:]] == [["r1", "T4", "0.5000"], ["r2", "T4", "1.2500"]]
