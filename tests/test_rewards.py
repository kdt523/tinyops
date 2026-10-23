import itertools
import json
import random
from types import SimpleNamespace

import pytest

from tinyops.policies.base import format_tool_call_text
from tinyops.policies.scripted import OraclePolicy
from tinyops.scenarios.build_dataset import load_scenarios
from tinyops.scoring.rewards import (
    DEFAULT_WEIGHTS,
    NAMES,
    RewardSet,
    components,
    replay,
    turn_texts,
)
from tinyops.sim.tools import ToolCall
from tinyops.utils.io import read_jsonl


@pytest.fixture(scope="module")
def val(built_data):
    return load_scenarios("val", built_data)


def pick(val, ftype, injected=False):
    return next(s for s in val if s.failure_type == ftype and s.injected == injected)


def oracle_texts(sc, seed=1):
    return [format_tool_call_text(c) for c in OraclePolicy(sc.state, random.Random(seed)).plan]


def traj(texts, final="Done."):
    """AgentTune-like Trajectory: one step per tool-call turn (thought = raw text) + final text."""
    steps = [SimpleNamespace(thought=t, action={"tool_calls": [1]}) for t in texts]
    return SimpleNamespace(steps=steps, final_response=final, metadata={})


def kw(sc_list, trajs):
    return {"scenario_id": [s.id for s in sc_list], "trajectories": trajs}


def total_of(rs, sc, texts, final="Done."):
    return rs.combined(None, **kw([sc], [traj(texts, final)]))[0]


def call(name, **args):
    return format_tool_call_text(ToolCall(name=name, arguments=args))


@pytest.fixture(scope="module")
def setup(val):
    sc = pick(val, "crashloop_missing_env")
    return sc, RewardSet({s.id: s for s in val})


def test_perfect_episode_components(setup):
    sc, rs = setup
    comp = rs.breakdown(kw([sc], [traj(oracle_texts(sc))]))[0]["components"]
    assert comp["diagnosis"] == 1.0 and comp["format"] == 0.2 and comp["schema"] == 0.0
    assert comp["efficiency"] == pytest.approx(0.1) and comp["safety"] == 0.0
    assert 0.25 <= comp["grounding"] <= 0.3
    expected = sum(DEFAULT_WEIGHTS[n] * comp[n] for n in NAMES)  # weighted sum of the components
    assert total_of(rs, sc, oracle_texts(sc)) == pytest.approx(expected)
    assert expected > 1.2


def test_correct_beats_malformed_beats_missing_three_different_scores(setup):
    """AgentTune's default reward setup scores these identically; ours must not."""
    sc, rs = setup
    good = total_of(rs, sc, oracle_texts(sc))
    malformed = rs.combined(None, **kw([sc], [traj([], final='[{"tool_name": "list_pods", "parameters": {"namespace": "x",}}]')]))[0]
    missing = rs.combined(None, **kw([sc], [traj([], final="I think the pods are crashing.")]))[0]
    assert good > malformed > missing
    assert len({round(good, 6), round(malformed, 6), round(missing, 6)}) == 3


def test_keyword_stuffed_instant_submit_loses_to_a_real_investigation(setup):
    sc, rs = setup
    kws = sc.state.ground_truth.root_cause_keywords
    stuffed_root = f"environment variable {kws[1][0]} is not set"
    real = total_of(rs, sc, oracle_texts(sc))
    for evidence in ([], ["made up evidence line"], [sc.alert_text]):
        stuffed = call("submit_diagnosis", root_cause=stuffed_root, evidence=evidence or ["x"],
                       recommended_fix="set it")
        t = total_of(rs, sc, [stuffed])
        assert t < real - 0.5, (evidence, t, real)
    comp = rs.breakdown(kw([sc], [traj([call("submit_diagnosis", root_cause=stuffed_root,
                                              evidence=["made up evidence line"], recommended_fix="x")])]))[0]["components"]
    assert comp["diagnosis"] == 0.3 and comp["grounding"] == 0.0  # right words, no grounding


def test_alert_text_is_not_grounded_evidence(setup):
    sc, rs = setup
    t = [call("submit_diagnosis", root_cause="environment variable " + sc.state.ground_truth.root_cause_keywords[1][0] + " unset",
              evidence=[sc.alert_text], recommended_fix="x")]
    assert rs.breakdown(kw([sc], [traj(t)]))[0]["components"]["grounding"] == 0.0


def test_delete_namespace_scores_below_every_non_unsafe_completion(setup, val):
    sc, rs = setup
    ns = sc.state.namespaces[0]
    texts = oracle_texts(sc)
    unsafe_perfect = total_of(rs, sc, [call("delete_namespace", namespace=ns), *texts])
    unsafe_alone = total_of(rs, sc, [call("delete_namespace", namespace=ns)])
    restart = call("restart_deployment", namespace=ns, deployment=sc.state.deployments[0].name, reason="r")
    safe = {
        "correct": total_of(rs, sc, texts),
        "prose": total_of(rs, sc, [], "no idea"),
        "malformed": total_of(rs, sc, [], '[{"tool_name": }]'),
        "loop": total_of(rs, sc, [call("list_pods", namespace=ns)] * 8),
        "wrong_but_clean": total_of(rs, sc, [call("list_pods", namespace=ns),
                                             call("submit_diagnosis", root_cause="pods are unhealthy", evidence=["NAME"], recommended_fix="x")]),
        "hallucinated": total_of(rs, sc, [call("kubectl_exec", cmd="ls")] * 3),
        "one_unnecessary_restart": total_of(rs, sc, [restart, *texts]),
    }
    assert unsafe_perfect < min(safe.values()), (unsafe_perfect, safe)
    assert unsafe_alone < unsafe_perfect
    comp = rs.breakdown(kw([sc], [traj([call("delete_namespace", namespace=ns), *texts])]))[0]["components"]
    assert comp["safety"] == -2.0 and comp["diagnosis"] == 1.0  # diagnosis cannot buy back a delete


def test_redundant_calls_lower_the_score_monotonically(setup):
    sc, rs = setup
    texts = oracle_texts(sc)
    first = texts[0]
    scores = [total_of(rs, sc, [first] * (1 + k) + texts[1:]) for k in range(4)]
    assert all(a > b for a, b in itertools.pairwise(scores)), scores


def test_correct_restart_on_hung_beats_diagnosis_alone(val, setup):
    _, rs = setup
    hung = pick(val, "hung_process_restart")
    full = oracle_texts(hung)
    no_restart = [t for t in full if "restart_deployment" not in t]
    assert len(no_restart) == len(full) - 1
    with_fix = rs.breakdown(kw([hung], [traj(full)]))[0]
    without = rs.breakdown(kw([hung], [traj(no_restart)]))[0]
    assert with_fix["components"]["safety"] == 0.2 and without["components"]["safety"] == 0.0
    assert with_fix["total"] > without["total"]


def test_restart_on_other_types_is_penalised(val, setup):
    _, rs = setup
    sc = pick(val, "oom_killed")
    texts = oracle_texts(sc)
    ns = sc.state.deployments[0].namespace
    dep = next(d.name for d in sc.state.deployments if d.namespace == ns)
    bad = [call("restart_deployment", namespace=ns, deployment=dep, reason="try"), *texts]
    base = rs.breakdown(kw([sc], [traj(texts)]))[0]
    pen = rs.breakdown(kw([sc], [traj(bad)]))[0]
    assert pen["components"]["safety"] == -0.5 and pen["total"] < base["total"] - 0.4


def test_schema_errors_and_hallucinated_tools(setup):
    sc, rs = setup
    ns = sc.state.namespaces[0]
    c = rs.breakdown(kw([sc], [traj([call("get_logs", namespace=ns),  # missing 'pod'
                                      call("kubectl_exec", cmd="x"),
                                      call("list_pods", namespace=ns)])]))[0]["components"]
    assert c["schema"] == pytest.approx(-0.5)
    many = traj([call("kubectl_exec", cmd=str(i)) for i in range(6)])
    assert rs.breakdown(kw([sc], [many]))[0]["components"]["schema"] == -1.0  # floor


def test_prose_costs_more_than_a_broken_call_and_format_is_floored(setup):
    sc, rs = setup
    prose = rs.breakdown(kw([sc], [traj([], "hello")]))[0]["components"]["format"]
    broken = rs.breakdown(kw([sc], [traj([], "[{\"tool_name\": }")]))[0]["components"]["format"]
    assert (prose, broken) == (-0.3, -0.2)
    assert rs.breakdown(kw([sc], [traj([], "")]))[0]["components"]["format"] == -0.3


def test_max_steps_hit_is_penalised_and_efficiency_bonus_needs_a_submit(setup):
    sc, rs = setup
    ns = sc.state.namespaces[0]
    loop = [call("get_events", namespace=ns, object_name=f"o{i}") for i in range(9)]
    c = rs.breakdown(kw([sc], [traj(loop)]))[0]["components"]
    assert c["efficiency"] == pytest.approx(-0.3)
    assert replay(loop, sc).episode.termination == "max_steps"


def test_tool_calls_after_a_valid_submit_are_discouraged_but_do_not_change_the_diagnosis(setup):
    sc, rs = setup
    texts = oracle_texts(sc)
    base = rs.breakdown(kw([sc], [traj(texts)]))[0]["components"]
    extra = rs.breakdown(kw([sc], [traj(texts + [call("list_pods", namespace="x")])]))[0]["components"]
    assert extra["diagnosis"] == base["diagnosis"] and extra["safety"] == base["safety"]
    assert extra["efficiency"] == pytest.approx(base["efficiency"] - 0.1)
    assert replay(texts + ["Done."], sc).post_submit_calls == 0


def test_conversations_fallback_when_no_trajectories(setup):
    sc, rs = setup
    conv = [{"role": "system", "content": "s"}, {"role": "user", "content": sc.alert_text}]
    for t in oracle_texts(sc):
        conv += [{"role": "assistant", "content": t}, {"role": "tool", "content": "x"}]
    conv.append({"role": "assistant", "content": "Done."})
    got = rs.combined(None, scenario_id=[sc.id], conversations=[conv])[0]
    assert got == pytest.approx(total_of(rs, sc, oracle_texts(sc)))
    assert turn_texts({"conversations": [conv]}, 0)[-1] == "Done."
    with pytest.raises(KeyError):
        turn_texts({}, 0)


def test_component_functions_follow_the_reward_signature_and_batch(setup, val):
    sc, rs = setup
    other = pick(val, "oom_killed")
    kws = kw([sc, other, sc], [traj(oracle_texts(sc)), traj([], "prose"), traj(oracle_texts(sc))])
    for name in NAMES:
        fn = rs.component(name)
        out = fn(["c1", "c2", "c3"], prompts=["p"] * 3, **kws)
        assert len(out) == 3 and all(isinstance(x, float) for x in out) and fn.__name__ == f"r_{name}"
    combined = rs.combined(["c"] * 3, **kws)
    assert combined[0] == combined[2] > combined[1]
    parts = [rs.component(n)(None, **kws)[0] for n in NAMES]
    assert combined[0] == pytest.approx(sum(DEFAULT_WEIGHTS[n] * p for n, p in zip(NAMES, parts, strict=True)))
    assert rs.reward_funcs() == [rs.combined]


def test_weights_and_ranges(setup, val):
    sc, _ = setup
    rs = RewardSet({s.id: s for s in val}, weights={"diagnosis": 2.0, "safety": 0.0})
    assert rs.weights["diagnosis"] == 2.0 and rs.weights["format"] == 0.5
    comp = components(replay(oracle_texts(sc), sc), sc)
    assert -0.6 <= comp["format"] <= 0.2 and -1.0 <= comp["schema"] <= 0 and -1.0 <= comp["efficiency"] <= 0.1
    assert 0 <= comp["grounding"] <= 0.3 and -2.5 <= comp["safety"] <= 0.2 and 0 <= comp["diagnosis"] <= 1


def test_logging_breakdown_and_trl_metrics(setup, val, tmp_path):
    sc, _ = setup
    log = tmp_path / "r.jsonl"
    rs = RewardSet({s.id: s for s in val}, log_path=log)
    metrics = {}
    rs.combined(None, log_metric=lambda k, v: metrics.__setitem__(k, v), **kw([sc], [traj(oracle_texts(sc))]))
    rows = read_jsonl(log)
    assert rows[0]["scenario_id"] == sc.id and rows[0]["diagnosis"] == 1.0 and len(rows[0]["texts"]) == 5  # 4 turns + the final plain-text turn
    assert set(metrics) == {f"reward/{n}" for n in NAMES}
    json.dumps(rows)


def test_replay_of_nothing_and_unknown_scenario(setup):
    sc, rs = setup
    rp = replay([], sc)
    assert rp.episode.termination == "policy_error" and rp.episode.steps == []
    with pytest.raises(KeyError):
        rs.combined(None, scenario_id=["nope"], trajectories=[traj([])])


def test_scores_are_cached_and_deterministic(setup):
    sc, rs = setup
    k = kw([sc], [traj(oracle_texts(sc))])
    a = rs.combined(None, **k)
    n = len(rs._cache)
    assert rs.combined(None, **k) == a and len(rs._cache) == n
