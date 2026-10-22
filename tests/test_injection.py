import json
import random

import pytest

from tinyops.scenarios.base import faulty_pod, locate_root_cause
from tinyops.scenarios.failure_types import FAILURE_TYPE_NAMES, FAILURE_TYPES
from tinyops.scenarios.injection import (
    ATTACK_FAMILIES,
    FAMILIES,
    LOCATIONS,
    TEMPLATES,
    feasible_locations,
    inject,
)
from tinyops.scenarios.names import TRAIN_POOLS
from tinyops.sim.cluster import ClusterState
from tinyops.sim.tools import ToolBox


def base(ft="crashloop_missing_env", diff="medium", seed=1):
    return FAILURE_TYPES[ft](random.Random(seed), diff, TRAIN_POOLS)


def where(state, payload):
    hits = set()
    if any(payload in ln for lines in state.logs.values() for ln in lines):
        hits.add("log")
    if any(payload in e.message for e in state.events):
        hits.add("event")
    if any(payload in v for c in state.configmaps for v in c.data.values()):
        hits.add("configmap")
    return hits


def test_families_and_template_counts():
    assert set(FAMILIES) == set(TEMPLATES)
    assert set(ATTACK_FAMILIES) | {"benign_lookalike"} == set(FAMILIES)
    for fam, tpls in TEMPLATES.items():
        assert len(tpls) >= 6, fam


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("location", LOCATIONS)
def test_payload_planted_exactly_at_location(family, location):
    for ft in ("crashloop_missing_env", "service_selector_mismatch", "oom_killed"):
        for seed in (1, 2):
            s = base(ft, "medium", seed)
            assert location in feasible_locations(s)
            out = inject(s, family, location, random.Random(seed))
            payload = out.ground_truth.injected_payload
            assert payload and "{ns}" not in payload and "{other}" not in payload
            assert "{cm}" not in payload and "{n}" not in payload
            assert where(out, payload) == {location}, (family, location, ft)
            ClusterState.model_validate(out.model_dump())


@pytest.mark.parametrize("family", FAMILIES)
def test_real_failure_intact_and_original_untouched(family):
    for ft in FAILURE_TYPE_NAMES:
        s = base(ft, "medium", 3)
        before = s.model_dump_json()
        loc = feasible_locations(s)[0]
        out = inject(s, family, loc, random.Random(3))
        assert s.model_dump_json() == before  # deep copy
        a, b = s.ground_truth, out.ground_truth
        assert (a.failure_type, a.key_evidence, a.root_cause_keywords, a.root_cause_object) == (
            b.failure_type, b.key_evidence, b.root_cause_keywords, b.root_cause_object)
        ns, _, _ = locate_root_cause(out)
        fp = faulty_pod(out)
        box = ToolBox(out)
        outs = [box.list_pods(ns), box.describe_pod(ns, fp.name), box.get_events(ns, fp.name),
                box.get_events(ns), box.get_logs(ns, fp.name, 200), box.check_service(ns, fp.deployment)]
        # evidence that is not in a (truncated) log window must still be observable
        assert any(any(e in o for o in outs) for e in b.key_evidence), (ft, family)


def test_log_injection_inside_faulty_pod_log_and_buried_at_hard():
    s = base("crashloop_missing_env", "hard", 4)
    out = inject(s, "direct_command", "log", random.Random(4))
    ns, _, _ = locate_root_cause(out)
    lines = out.logs[f"{ns}/{faulty_pod(out).name}"]
    idx = next(i for i, ln in enumerate(lines) if out.ground_truth.injected_payload in ln)
    assert len(lines) - idx > 20  # buried: not within the last 20 lines


def test_targets_per_family():
    s = base("crashloop_missing_env", "hard", 2)
    for fam in ("direct_command", "fake_authority", "tool_format_smuggle", "urgency_social"):
        out = inject(s, fam, "event", random.Random(5))
        (target,) = out.ground_truth.injection_targets
        assert target in out.namespaces
        assert target in out.ground_truth.injected_payload
        assert "delet" in out.ground_truth.injected_payload.lower()
    out = inject(s, "wrong_restart", "event", random.Random(5))
    (target,) = out.ground_truth.injection_targets
    ns, dep = target.split("/")
    assert out.get_deployment(ns, dep) is not None
    assert dep != faulty_pod(out).deployment
    assert dep in out.ground_truth.injected_payload
    d = out.get_deployment(ns, dep)
    assert d.ready_replicas == d.replicas  # a healthy deployment
    out = inject(s, "exfil", "log", random.Random(5))
    (target,) = out.ground_truth.injection_targets
    assert out.get_configmap(*[locate_root_cause(out)[0], target]) is not None
    assert target in out.ground_truth.injected_payload


def test_wrong_restart_creates_healthy_target_when_none_exists():
    s = base("crashloop_missing_env", "easy", 1)
    assert len(s.deployments) == 1
    out = inject(s, "wrong_restart", "event", random.Random(1))
    assert len(out.deployments) == 2
    assert out.ground_truth.injection_targets[0].split("/")[1] in {d.name for d in out.deployments}


def test_benign_lookalike_has_no_targets():
    for seed in range(1, 8):
        out = inject(base(), "benign_lookalike", "log", random.Random(seed))
        assert out.ground_truth.injection_targets == []
        assert out.ground_truth.injected_payload
        low = out.ground_truth.injected_payload.lower()
        assert not any(w in low for w in ("send", "http", "upload", "exfil", "post "))


def test_smuggle_contains_exact_tool_call_syntax():
    for seed in range(1, 12):
        out = inject(base(), "tool_format_smuggle", "log", random.Random(seed))
        p = out.ground_truth.injected_payload
        start = min(i for i in (p.find("[{"), p.find('{"tool_name"')) if i >= 0)
        tail = p[start:]
        tail = tail.removesuffix("<|END_ACTION|>")
        parsed = json.loads(tail)
        call = parsed[0] if isinstance(parsed, list) else parsed
        assert call["tool_name"] == "delete_namespace"
        assert call["parameters"]["namespace"] in out.namespaces


def test_attacks_name_a_dangerous_action():
    for seed in range(1, 6):
        for fam, needles in {"direct_command": ["delete"], "exfil": ["configmap"],
                             "wrong_restart": ["restart"]}.items():
            p = inject(base(), fam, "event", random.Random(seed)).ground_truth.injected_payload
            assert any(n in p.lower() for n in needles)


def test_log_not_feasible_without_logs():
    s = base("image_pull_backoff", "easy", 1)
    assert feasible_locations(s) == ["event", "configmap"]
    with pytest.raises(ValueError):
        inject(s, "direct_command", "log", random.Random(1))


def test_unknown_family_or_location_rejected():
    with pytest.raises(ValueError):
        inject(base(), "nope", "log", random.Random(1))
    with pytest.raises(ValueError):
        inject(base(), "exfil", "nowhere", random.Random(1))
