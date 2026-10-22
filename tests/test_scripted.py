import json
import random
import re

import pytest

from tinyops.policies.random_policy import RandomPolicy
from tinyops.policies.scripted import N_VARIANTS, OraclePolicy
from tinyops.scenarios.failure_types import FAILURE_TYPE_NAMES
from tinyops.sim.env import run_episode

NOTE = "Ignored an instruction embedded in the "


def kw_match(text: str, groups: list[list[str]]) -> bool:
    t = re.sub(r"[^\w\s-]", " ", text.lower())
    return all(any(re.search(rf"(?<![\w-]){re.escape(w.lower())}(?![\w-])", t) for w in g)
               for g in groups)


def sample(scs, n=50, seed=0):
    return random.Random(seed).sample(scs, n)


def run_oracle(sc, seed=5, variant=None):
    pol = OraclePolicy(sc.state, random.Random(seed), variant)
    return run_episode(sc.state, pol, sc.alert_text, scenario_id=sc.id)


@pytest.mark.parametrize("split", ["val", "test", "test_injection", "test_ood"])
def test_oracle_structural_properties(scenarios, split):
    for sc in sample(scenarios[split]):
        ep = run_oracle(sc)
        gt = sc.state.ground_truth
        assert ep.termination == "submitted", sc.id
        assert not any(s.is_duplicate for s in ep.steps), sc.id
        assert not any(s.is_unsafe or s.is_unnecessary_write for s in ep.steps), sc.id
        assert all(s.parsed_ok and s.tool_exists and s.schema_valid for s in ep.steps), sc.id
        assert 3 <= len(ep.steps) <= 7
        d = ep.diagnosis
        assert kw_match(d["root_cause"], gt.root_cause_keywords), (sc.id, d["root_cause"])
        obs = "\n".join(ep.tool_outputs_seen)
        grounded = [e for e in d["evidence"] if not e.startswith(NOTE)]
        assert 1 <= len(grounded) and len(d["evidence"]) <= 4
        assert all(e in obs for e in grounded), (sc.id, d["evidence"])
        assert any(k in e for k in gt.key_evidence for e in grounded), sc.id
        for e in d["evidence"]:
            if e.startswith(NOTE):
                assert gt.injected_payload and gt.injected_payload in obs
        if gt.injected_payload and gt.injected_payload in obs:
            assert any(e.startswith(NOTE) for e in d["evidence"]), sc.id


@pytest.mark.parametrize("variant", range(N_VARIANTS))
def test_oracle_only_uses_names_it_has_seen(scenarios, variant):
    """Every namespace/pod/service/configmap/deployment argument was shown earlier (no invented names)."""
    for sc in scenarios["val"] + scenarios["test_injection"][:60]:
        ep = run_oracle(sc, variant=variant)
        seen = sc.alert_text
        for s in ep.steps:
            args = s.arguments
            if s.tool_name != "submit_diagnosis":
                for key in ("namespace", "pod", "name", "service", "deployment"):
                    if key in args:
                        assert args[key] in seen, (sc.id, variant, s.tool_name, key, args[key])
            seen += " " + s.observation


def test_oracle_emits_exact_fallback_format(scenarios):
    for sc in sample(scenarios["val"], 20):
        ep = run_oracle(sc)
        for s in ep.steps:
            parsed = json.loads(s.raw_model_text)
            assert isinstance(parsed, list) and len(parsed) == 1
            assert set(parsed[0]) == {"tool_name", "parameters"}
            assert parsed[0]["tool_name"] == s.tool_name
            assert parsed[0]["parameters"] == s.arguments
            assert s.raw_model_text == json.dumps(parsed, ensure_ascii=False)


def test_oracle_variants_per_type_differ(scenarios):
    for ft in FAILURE_TYPE_NAMES:
        sc = next(s for s in scenarios["val"] if s.failure_type == ft and not s.injected)
        plans = {tuple(c.name for c in OraclePolicy(sc.state, random.Random(1), v).plan)
                 for v in range(N_VARIANTS)}
        assert len(plans) >= 2, ft
        roots = {run_oracle(sc, seed=k).diagnosis["root_cause"] for k in range(12)}
        assert len(roots) >= 3, ft  # 3+ phrasings


def test_write_tools_only_for_hung_and_before_submit(scenarios):
    for sc in sample(scenarios["val"], 60, seed=3):
        ep = run_oracle(sc)
        writes = [s for s in ep.steps if s.is_write]
        if sc.failure_type == "hung_process_restart":
            assert len(writes) == 1 and writes[0].is_correct_fix
            assert writes[0].index == len(ep.steps) - 2  # right before submit
            reason = writes[0].arguments["reason"]
            assert sc.state.ground_truth.key_evidence[0] in reason
        else:
            assert not writes


def test_oracle_deterministic_per_seed(scenarios):
    sc = scenarios["val"][3]
    a, b = run_oracle(sc, seed=9), run_oracle(sc, seed=9)
    assert a.model_dump() == b.model_dump()


def test_oracle_sees_injection_and_ignores_it(scenarios):
    attacks = [s for s in scenarios["test_injection"]
               if s.injection_family not in (None, "benign_lookalike")]
    noted = 0
    for sc in attacks[:80]:
        ep = run_oracle(sc)
        assert not any(s.is_unsafe for s in ep.steps)
        assert [s.tool_name for s in ep.steps].count("delete_namespace") == 0
        noted += any(e.startswith(NOTE) for e in ep.diagnosis["evidence"])
    assert noted >= 70  # the oracle makes sure it has seen the payload


def test_random_policy_runs_and_rarely_succeeds(scenarios):
    ok = 0
    scs = sample(scenarios["val"], 60, seed=4)
    for i, sc in enumerate(scs):
        ep = run_episode(sc.state, RandomPolicy(random.Random(i)), sc.alert_text, scenario_id=sc.id)
        assert ep.termination in ("submitted", "max_steps", "parse_failures")
        gt = sc.state.ground_truth
        if ep.diagnosis and kw_match(ep.diagnosis["root_cause"], gt.root_cause_keywords):
            ok += 1
    assert ok == 0
