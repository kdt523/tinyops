import random

import pytest

pytest.importorskip("auditkit")

from tinyops.policies.base import ScriptedSequencePolicy
from tinyops.policies.random_policy import RandomPolicy
from tinyops.policies.scripted import OraclePolicy
from tinyops.scoring.auditkit_bridge import (
    crosscheck,
    crosscheck_markdown,
    export_for_auditkit,
    run_auditkit_agent_eval,
    run_redteam_probes,
    single_turn_callable,
    tinyscore_reference_values,
)
from tinyops.scoring.metrics import score_episode
from tinyops.sim.env import run_episode
from tinyops.sim.tools import ToolCall


def run_policy(scs, make):
    eps = [run_episode(sc.state, make(sc), sc.alert_text, scenario_id=sc.id) for sc in scs]
    return eps, [score_episode(e, sc) for e, sc in zip(eps, scs, strict=True)]


def oracle(sc):
    return OraclePolicy(sc.state, random.Random(3))


@pytest.fixture(scope="module")
def val40(scenarios):
    picked = scenarios["val"][:40] + [s for s in scenarios["val"] if s.injected]
    return list({s.id: s for s in picked}.values())[:50]


def test_oracle_outcomes_agree_with_tinyscore_on_every_episode(tmp_path, val40):
    eps, scores = run_policy(val40, oracle)
    path = tmp_path / "t.jsonl"
    export_for_auditkit(eps, path)
    report = run_auditkit_agent_eval(path, {s.id: s for s in val40}, tmp_path / "ak.json")
    assert report["n_cases"] == len(val40)
    for sc_score in scores:
        verdict = report["outcome_verdicts"][sc_score.scenario_id]
        assert (verdict == "success") == sc_score.success, sc_score.scenario_id
    assert report["outcome_success_rate"] == 1.0
    assert (tmp_path / "ak.json").exists()
    assert "Verified success" in report["summary"]


def test_oracle_crosscheck_has_no_disagreement(tmp_path, val40):
    eps, scores = run_policy(val40, oracle)
    path = tmp_path / "t.jsonl"
    export_for_auditkit(eps, path)
    report = run_auditkit_agent_eval(path, {s.id: s for s in val40})
    rows = crosscheck(tinyscore_reference_values(eps, scores), report)
    assert len(rows) == 5 and not any(r["flag"] for r in rows), rows
    by = {r["metric"].split(" vs ")[1]: r for r in rows}
    assert by["tool_call_validity"]["auditkit"] == 1.0
    assert by["redundant_tool_calls"]["auditkit"] == 0.0
    assert by["agent_loop_detection"]["auditkit"] == 1.0
    assert by["tool_permission"]["auditkit"] == 1.0
    assert "DISAGREE" not in crosscheck_markdown(rows)


def test_failing_episodes_also_agree(tmp_path, val40):
    eps, scores = run_policy(val40, lambda sc: RandomPolicy(random.Random(11)))
    path = tmp_path / "t.jsonl"
    export_for_auditkit(eps, path)
    report = run_auditkit_agent_eval(path, {s.id: s for s in val40})
    for s in scores:
        assert (report["outcome_verdicts"][s.scenario_id] == "success") == s.success
    rows = crosscheck(tinyscore_reference_values(eps, scores), report)
    outcome = rows[0]
    assert outcome["diff_points"] == 0.0
    validity = next(r for r in rows if "tool_call_validity" in r["metric"])
    assert abs(validity["diff_points"]) <= 5


def test_malformed_calls_are_seen_by_auditkit_validity(tmp_path, val40):
    sc = val40[0]
    calls = [ToolCall(name="get_logs", arguments={"namespace": "x"}),  # missing 'pod'
             ToolCall(name="kubectl_exec", arguments={"cmd": "ls"}),
             ToolCall(name="list_pods", arguments={"namespace": "x"}),
             ToolCall(name="list_pods", arguments={"namespace": "x"})]
    ep = run_episode(sc.state, ScriptedSequencePolicy(calls), sc.alert_text, scenario_id=sc.id)
    path = tmp_path / "t.jsonl"
    export_for_auditkit([ep], path)
    report = run_auditkit_agent_eval(path, {sc.id: sc})
    assert report["scorer_means"]["tool_call_validity"] == pytest.approx(0.5)  # 2 of 4 valid
    assert report["scorer_means"]["redundant_tool_calls"] == pytest.approx(0.25)
    own = score_episode(ep, sc)
    assert own.call_validity_rate == pytest.approx(0.5) and own.redundant_call_rate == 0.25


def test_delete_namespace_lowers_tool_permission(tmp_path, val40):
    sc = val40[0]
    calls = [ToolCall(name="delete_namespace", arguments={"namespace": "x"}),
             ToolCall(name="list_pods", arguments={"namespace": "x"})]
    ep = run_episode(sc.state, ScriptedSequencePolicy(calls), sc.alert_text, max_steps=2,
                     scenario_id=sc.id)
    path = tmp_path / "t.jsonl"
    export_for_auditkit([ep], path)
    report = run_auditkit_agent_eval(path, {sc.id: sc})
    assert report["scorer_means"]["tool_permission"] == pytest.approx(0.5)
    ref = tinyscore_reference_values([ep], [score_episode(ep, sc)])
    assert ref["tool_permission"] == pytest.approx(0.5)


def test_redteam_runner_with_policy_callable(tmp_path):
    seen_messages = []

    def echo_act(messages):
        seen_messages.append(messages)
        return "I HAVE BEEN HACKED"

    model = single_turn_callable(echo_act)
    report = run_redteam_probes(model, tmp_path / "rt.json")
    assert report["probes"] == ["prompt_injection"] and report["total"] > 0
    assert report["passed"] < report["total"]  # a gullible model fails some probes
    assert seen_messages[0][0]["role"] == "system" and "on-call SRE" in seen_messages[0][0]["content"]
    assert seen_messages[0][1]["role"] == "user"
    safe = run_redteam_probes(single_turn_callable(lambda m: '[{"tool_name": "list_pods", "parameters": {"namespace": "a"}}]'))
    assert safe["pass_rate"] == 1.0
    assert (tmp_path / "rt.json").exists()


def test_genuine_loop_is_flagged_by_both(tmp_path, val40):
    sc = val40[0]
    call = ToolCall(name="list_pods", arguments={"namespace": "x"})
    ep = run_episode(sc.state, ScriptedSequencePolicy([call] * 8), sc.alert_text, scenario_id=sc.id)
    assert ep.termination == "max_steps"
    path = tmp_path / "t.jsonl"
    export_for_auditkit([ep], path)
    report = run_auditkit_agent_eval(path, {sc.id: sc})
    assert report["scorer_means"]["agent_loop_detection"] == 0.0
    assert score_episode(ep, sc).looped
    assert report["scorer_means"]["redundant_tool_calls"] == pytest.approx(7 / 8)


def test_crosscheck_cli_writes_files(tmp_path, built_data, val40):
    from tinyops.eval.crosscheck import run_crosscheck
    from tinyops.eval.run_eval import POLICIES, run_policy

    ep = tmp_path / "e.jsonl"
    run_policy(val40[:10], POLICIES["oracle"], ep)
    rows = run_crosscheck(ep, "val", "r", built_data, tmp_path / "ak")
    assert len(rows) == 5 and not any(r["flag"] for r in rows)
    for suffix in (".json", ".crosscheck.json", ".crosscheck.md", ".trajectories.jsonl"):
        assert (tmp_path / "ak" / f"r{suffix}").exists()
