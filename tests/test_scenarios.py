import random
import re

import pytest

from tinyops.policies.scripted import _log_tail
from tinyops.scenarios import noise
from tinyops.scenarios.alerts import OOD_ALERTS, TRAIN_ALERTS, make_alert
from tinyops.scenarios.base import DIFFICULTIES, faulty_pod, locate_root_cause
from tinyops.scenarios.failure_types import FAILURE_TYPE_NAMES, FAILURE_TYPES, LOG_KEY_TYPES
from tinyops.scenarios.names import OOD_POOLS, TRAIN_POOLS
from tinyops.sim.cluster import ClusterState, parse_mem
from tinyops.sim.tools import ToolBox

CASES = [(t, d, seed) for t in FAILURE_TYPE_NAMES for d in DIFFICULTIES for seed in (1, 2)]


def gen(ft, diff, seed, pools=TRAIN_POOLS):
    return FAILURE_TYPES[ft](random.Random(seed), diff, pools)


def matches(text: str, groups: list[list[str]]) -> bool:
    t = re.sub(r"[^\w\s-]", " ", text.lower())
    return all(any(re.search(rf"(?<![\w-]){re.escape(w.lower())}(?![\w-])", t) for w in g) for g in groups)


def observe(state: ClusterState) -> list[str]:
    """Outputs of the relevant tools called with the right args."""
    ns, _, _ = locate_root_cause(state)
    fp = faulty_pod(state)
    box = ToolBox(state)
    gt = state.ground_truth
    frags = [e for e in gt.key_evidence if any(e in ln for ln in state.logs.get(f"{ns}/{fp.name}", []))]
    tail = _log_tail(state, ns, fp.name, frags)
    cm = fp.env_from_configmap or ""
    outs = [
        box.list_pods(ns), box.describe_pod(ns, fp.name), box.get_events(ns),
        box.get_events(ns, fp.name), box.get_logs(ns, fp.name, tail),
        box.get_metrics(ns, fp.name), box.check_service(ns, fp.deployment or ""),
        box.get_configmap(ns, cm),
    ]
    return outs


@pytest.mark.parametrize(("ft", "diff", "seed"), CASES)
def test_state_validates_and_evidence_observable(ft, diff, seed):
    s = gen(ft, diff, seed)
    ClusterState.model_validate(s.model_dump())
    gt = s.ground_truth
    assert gt.failure_type == ft and gt.key_evidence and gt.root_cause_keywords
    assert "list_pods" in gt.relevant_tools
    outs = observe(s)
    for frag in gt.key_evidence:
        assert any(frag in o for o in outs), (ft, diff, frag)
    # root cause object really exists
    locate_root_cause(s)
    assert faulty_pod(s).name


@pytest.mark.parametrize(("ft", "diff", "seed"), CASES)
def test_alert_never_states_root_cause(ft, diff, seed):
    s = gen(ft, diff, seed)
    gt = s.ground_truth
    for ood in (False, True):
        for _ in range(12):
            alert = make_alert(random.Random(seed * 100 + _), s, ood=ood)
            assert not matches(alert, gt.root_cause_keywords), alert
            assert not any(e in alert for e in gt.key_evidence), alert


def test_every_alert_template_is_usable_and_formatted():
    for ft in FAILURE_TYPE_NAMES:
        gen(ft, "medium", 3)
        for pool in (TRAIN_ALERTS, OOD_ALERTS):
            for t in pool:
                if t.types is None or ft in t.types:
                    assert "{" not in t.text.format(app="a", ns="n", pod="p")
    assert len(TRAIN_ALERTS) >= 8 and len(OOD_ALERTS) >= 4
    assert all(("KubePodCrashLooping" not in t.text) or t.types for t in TRAIN_ALERTS)


@pytest.mark.parametrize("ft", FAILURE_TYPE_NAMES)
def test_deterministic_for_seed(ft):
    a = gen(ft, "hard", 7).model_dump_json()
    assert a == gen(ft, "hard", 7).model_dump_json()
    assert a != gen(ft, "hard", 8).model_dump_json()


@pytest.mark.parametrize("ft", FAILURE_TYPE_NAMES)
def test_difficulty_knobs(ft):
    for seed in (1, 2, 3):
        easy, med, hard = (gen(ft, d, seed) for d in DIFFICULTIES)
        assert len(easy.namespaces) == 1 and len(easy.deployments) == 1
        assert len(med.namespaces) == 1 and len(med.deployments) >= 3
        assert len(hard.namespaces) >= 2 and len(hard.deployments) >= 4
        assert any(p.phase == "Completed" for p in hard.pods)
        assert not any(p.phase == "Completed" for p in easy.pods + med.pods)
        # unrelated warning events at hard
        assert any(e.type == "Warning" and e.namespace != locate_root_cause(hard)[0]
                   for e in hard.events)


@pytest.mark.parametrize("ft", FAILURE_TYPE_NAMES)
def test_log_lengths_follow_difficulty(ft):
    ranges = {"easy": (10, 15), "medium": (20, 40), "hard": (60, 120)}
    for diff, (lo, hi) in ranges.items():
        for seed in (1, 2, 3):
            s = gen(ft, diff, seed)
            ns, _, _ = locate_root_cause(s)
            fp = faulty_pod(s)
            lines = s.logs[f"{ns}/{fp.name}"]
            if lines:  # image/pending/eviction pods have no logs
                assert lo <= len(lines) <= hi, (ft, diff, len(lines))


@pytest.mark.parametrize("ft", sorted(LOG_KEY_TYPES))
def test_hard_buries_key_log_line_outside_last_20(ft):
    for seed in range(1, 8):
        s = gen(ft, "hard", seed)
        ns, _, _ = locate_root_cause(s)
        lines = s.logs[f"{ns}/{faulty_pod(s).name}"]
        positions = [len(lines) - i for i, ln in enumerate(lines)
                     if any(e in ln for e in s.ground_truth.key_evidence)]
        assert positions, ft
        assert min(positions) > 20, (ft, seed, positions)


def test_hard_has_misleading_warning():
    found = 0
    for ft in ("crashloop_missing_env", "oom_killed", "hung_process_restart"):
        for seed in range(1, 5):
            s = gen(ft, "hard", seed)
            ns, _, _ = locate_root_cause(s)
            lines = s.logs[f"{ns}/{faulty_pod(s).name}"]
            found += any("latency" in ln or "slowly" in ln for ln in lines)
    assert found >= 8


def test_noise_has_no_key_evidence_substrings():
    rng = random.Random(0)
    for _ in range(300):
        for ln in noise.healthy_logs(rng, "app", 30, "svc"):
            assert not any(bad in ln for bad in noise.FORBIDDEN_IN_NOISE), ln
    assert len({noise.benign_line(random.Random(i), "x").split()[0] for i in range(50)}) >= 3


def test_pod_names_follow_pattern():
    pat = re.compile(r"^[a-z0-9-]+-[0-9a-f]{6}-[a-z0-9]{5}$")
    for ft in FAILURE_TYPE_NAMES:
        for p in gen(ft, "hard", 2).pods:
            if p.deployment:
                assert pat.match(p.name), p.name


# -- per-type consistency between logs, events, describe output and metrics --------------------
def test_crashloop_state_consistent():
    s = gen("crashloop_missing_env", "medium", 4)
    var = s.ground_truth.root_cause_keywords[1][0]
    fp = faulty_pod(s)
    assert fp.phase == "CrashLoopBackOff" and not fp.ready and 5 <= fp.restarts <= 40
    assert fp.last_state.exit_code == 1 and var not in fp.env
    box = ToolBox(s)
    d = box.describe_pod(fp.namespace, fp.name)
    assert "Reason: CrashLoopBackOff" in d and "Exit Code: 1" in d and var not in d
    assert "Back-off restarting failed container" in box.get_events(fp.namespace, fp.name)


def test_oom_state_consistent():
    for seed in range(1, 6):
        s = gen("oom_killed", "medium", seed)
        for p in s.pods:
            if p.deployment == faulty_pod(s).deployment:
                m = s.metrics[f"{p.namespace}/{p.name}"]
                assert m.mem_bytes >= 0.95 * parse_mem(p.resources.mem_limit)
                assert p.last_state.reason == "OOMKilled" and p.last_state.exit_code == 137
        d = ToolBox(s).describe_pod(faulty_pod(s).namespace, faulty_pod(s).name)
        assert "OOMKilled" in d and "Exit Code: 137" in d
        assert "memory usage: 9" in ToolBox(s).get_metrics(faulty_pod(s).namespace, faulty_pod(s).name)


def test_image_pull_consistent():
    s = gen("image_pull_backoff", "easy", 2)
    fp = faulty_pod(s)
    dep = s.get_deployment(fp.namespace, fp.deployment)
    assert fp.image == dep.image and fp.phase == "ImagePullBackOff"
    assert s.logs[f"{fp.namespace}/{fp.name}"] == []
    assert ToolBox(s).get_logs(fp.namespace, fp.name) == "(no logs)"
    ev = ToolBox(s).get_events(fp.namespace, fp.name)
    assert f'Failed to pull image "{fp.image}"' in ev and "not found" in ev


def test_readiness_variants_both_occur_and_are_consistent():
    kinds = set()
    for seed in range(1, 12):
        s = gen("readiness_probe_misconfig", "easy", seed)
        fp = faulty_pod(s)
        assert fp.phase == "Running" and not fp.ready and fp.restarts == 0
        probe = fp.probes[0]
        assert not probe.passing
        ev = ToolBox(s).get_events(fp.namespace, fp.name)
        if "connection refused" in ev:
            kinds.add("port")
            assert probe.port != int(fp.env["PORT"])
        else:
            kinds.add("path")
            assert "statuscode: 404" in ev
    assert kinds == {"port", "path"}


def test_selector_mismatch_consistent():
    for seed in range(1, 8):
        s = gen("service_selector_mismatch", "medium", seed)
        fp = faulty_pod(s)
        assert fp.ready and fp.phase == "Running"
        svc = s.get_service(fp.namespace, fp.deployment)
        assert not s.pods_matching(fp.namespace, svc.selector)
        out = ToolBox(s).check_service(fp.namespace, svc.name)
        assert "Endpoints: <none>" in out
    assert s.ground_truth.root_cause_object == f"service/{fp.deployment}"


def test_pending_consistent():
    s = gen("pending_insufficient_memory", "medium", 3)
    fp = faulty_pod(s)
    free = [parse_mem(n.allocatable_mem) - parse_mem(n.allocated_mem) for n in s.nodes]
    assert fp.phase == "Pending" and fp.node is None
    assert all(parse_mem(fp.resources.mem_request) > f for f in free)
    assert "0/3 nodes are available: 3 Insufficient memory" in ToolBox(s).get_events(fp.namespace)
    assert ToolBox(s).get_metrics(fp.namespace, fp.name) == "Error: metrics not available yet"


def test_eviction_consistent():
    s = gen("node_memory_pressure_eviction", "hard", 3)
    fp = faulty_pod(s)
    assert fp.phase == "Evicted"
    node = next(n for n in s.nodes if n.name == fp.node)
    assert node.memory_pressure
    assert "The node was low on resource: memory" in ToolBox(s).get_events(fp.namespace)


def test_bad_configmap_consistent():
    for seed in range(1, 8):
        s = gen("bad_configmap_value", "medium", seed)
        fp = faulty_pod(s)
        key = s.ground_truth.root_cause_keywords[1][0]
        cm = s.get_configmap(fp.namespace, fp.env_from_configmap)
        assert key in cm.data
        assert f"{key}: {cm.data[key]}" in ToolBox(s).get_configmap(fp.namespace, cm.name)
        assert s.get_deployment(fp.namespace, fp.deployment).configmap == cm.name


def test_hung_consistent_and_has_fix():
    s = gen("hung_process_restart", "medium", 5)
    fp = faulty_pod(s)
    assert fp.phase == "Running" and not fp.ready
    assert not any(p.kind == "liveness" for p in fp.probes)
    gt = s.ground_truth
    assert gt.correct_fix.tool == "restart_deployment"
    assert gt.correct_fix.arguments == {"namespace": fp.namespace, "deployment": fp.deployment}
    assert s.metrics[f"{fp.namespace}/{fp.name}"].cpu_millicores <= 2
    if fp.restarts:
        assert fp.last_state is not None


def test_only_hung_has_a_correct_fix():
    for ft in FAILURE_TYPE_NAMES:
        fix = gen(ft, "easy", 1).ground_truth.correct_fix
        assert (fix is not None) == (ft == "hung_process_restart")


def test_ood_pools_do_not_overlap_train():
    for field in ("namespaces", "apps", "env_vars", "int_config_keys", "url_config_keys", "nodes",
                  "cron_jobs", "downstream"):
        assert not set(getattr(TRAIN_POOLS, field)) & set(getattr(OOD_POOLS, field)), field
    assert TRAIN_POOLS.registry != OOD_POOLS.registry
    assert len(TRAIN_POOLS.namespaces) >= 20 and len(TRAIN_POOLS.apps) >= 30
    assert not set(TRAIN_POOLS.ports) & set(OOD_POOLS.ports)


def test_ood_scenarios_use_ood_names_only():
    for ft in FAILURE_TYPE_NAMES:
        s = gen(ft, "medium", 2, OOD_POOLS)
        assert set(s.namespaces) <= set(OOD_POOLS.namespaces)
        assert {d.name for d in s.deployments} <= set(OOD_POOLS.apps)
        assert all(p.image.startswith("ghcr.io/initech/") for p in s.pods if p.deployment)
