"""The 9 failure generators. Each keeps logs, events, describe output and metrics consistent."""

from __future__ import annotations

import random
from collections.abc import Callable

from tinyops.scenarios.base import World, shape_logs
from tinyops.scenarios.names import BAD_INT_VALUES, BAD_URL_VALUES, NamePools
from tinyops.sim.cluster import (
    ClusterState,
    ConfigMap,
    ContainerState,
    Event,
    Fix,
    GroundTruth,
    Pod,
    PodMetrics,
    Probe,
    parse_mem,
)
from tinyops.sim.render import pod_ip

BASE_TOOLS = ["list_pods", "describe_pod"]


def _setup(rng: random.Random, difficulty: str, pools: NamePools):
    """World with the faulty deployment and the difficulty-dependent extras."""
    w = World(rng, pools, difficulty)
    ns = w.new_namespace()
    app = w.new_app()
    replicas = rng.randint(1, 2) if difficulty == "easy" else rng.randint(2, 3)
    log_lines = {"easy": (10, 15), "medium": (20, 40), "hard": (60, 120)}[difficulty]
    dep, pods = w.add_deployment(ns, app, replicas, log_lines=log_lines)
    if difficulty != "easy":
        w.add_healthy_extras(ns, rng.randint(2, 3))
    if difficulty == "hard":
        w.add_hard_distractors(w.new_namespace(exclude=(ns,)))
    pods.sort(key=lambda p: p.name)
    return w, ns, app, dep, pods


def _downstream(w: World) -> str:
    return w.rng.choice(w.pools.downstream)


def _set_logs(w: World, pods: list[Pod], key_lines: list[str], *, after_fn=None) -> None:
    """Give each faulty pod its own noisy log with `key_lines` placed per difficulty."""
    for p in pods:
        lines, _ = shape_logs(
            w.rng, w.difficulty, p.deployment or "app", _downstream(w), key_lines,
            after_lines_fn=after_fn, misleading=w.difficulty == "hard",
        )
        w.logs[f"{p.namespace}/{p.name}"] = lines


def _event(w: World, pod: Pod, type_: str, reason: str, message: str, age: str, count: int = 1,
           kind: str = "Pod", name: str | None = None) -> None:
    w.events.append(
        Event(type=type_, reason=reason, object_kind=kind, object_name=name or pod.name,
              namespace=pod.namespace, message=message, count=count, age=age)
    )


def _crash(w: World, pods: list[Pod], dep, reason: str = "Error", exit_code: int = 1) -> None:
    rng = w.rng
    for p in pods:
        p.phase, p.ready = "CrashLoopBackOff", False
        p.restarts = rng.randint(5, 40)
        p.current_state = ContainerState(state="waiting", reason="CrashLoopBackOff")
        p.last_state = ContainerState(state="terminated", reason=reason, exit_code=exit_code)
        p.probes = [Probe(kind="readiness", path="/healthz", port=pr.port, passing=False)
                    for pr in p.probes]
        w.metrics[f"{p.namespace}/{p.name}"] = PodMetrics(
            cpu_millicores=rng.randint(0, 5), mem_bytes=rng.randint(20, 60) * 1024**2)
        _event(w, p, "Warning", "BackOff", "Back-off restarting failed container",
               f"{rng.randint(1, 3)}m", count=rng.randint(10, 60))
    dep.ready_replicas = 0


def _gt(failure_type: str, pods: list[Pod], keywords: list[list[str]], evidence: list[str],
        tools: list[str], fix: Fix | None = None, obj: str | None = None) -> GroundTruth:
    return GroundTruth(
        failure_type=failure_type, root_cause_object=obj or f"pod/{pods[0].name}",
        root_cause_keywords=keywords, key_evidence=evidence, relevant_tools=tools, correct_fix=fix,
    )


# --------------------------------------------------------------------------------------------
def gen_crashloop_missing_env(rng: random.Random, difficulty: str, pools: NamePools) -> ClusterState:
    w, _ns, _app, dep, pods = _setup(rng, difficulty, pools)
    var = rng.choice(pools.env_vars)
    if rng.random() < 0.5:
        frag = f"KeyError: '{var}'"
        key_lines = ["Traceback (most recent call last):",
                     '  File "app/main.py", line 12, in <module>',
                     f"    url = os.environ['{var}']", frag]
    else:
        frag = f"required environment variable {var} is not set"
        key_lines = [f"ERROR config: {frag}", "FATAL: aborting startup"]
    others = [v for v in pools.env_vars if v != var]
    for p in pods:
        p.env.update({k: f"{k.lower()}-value" for k in rng.sample(others, 2)})
    _crash(w, pods, dep)
    _set_logs(w, pods, key_lines)
    gt = _gt("crashloop_missing_env", pods, [["env", "environment variable", "variable"], [var]],
             [frag], [*BASE_TOOLS, "get_logs"])
    return w.state(gt)


def gen_oom_killed(rng: random.Random, difficulty: str, pools: NamePools) -> ClusterState:
    w, _ns, _app, dep, pods = _setup(rng, difficulty, pools)
    crash_state = rng.random() < 0.5 or difficulty == "easy"
    for p in pods:
        p.restarts = rng.randint(3, 25)
        p.last_state = ContainerState(state="terminated", reason="OOMKilled", exit_code=137)
        if crash_state:
            p.phase, p.ready = "CrashLoopBackOff", False
            p.current_state = ContainerState(state="waiting", reason="CrashLoopBackOff")
        limit = parse_mem(p.resources.mem_limit)
        w.metrics[f"{p.namespace}/{p.name}"] = PodMetrics(
            cpu_millicores=rng.randint(100, 450), mem_bytes=int(limit * rng.uniform(0.95, 0.99)))
        _event(w, p, "Normal", "Started", "Started container app", f"{rng.randint(1, 4)}m")
        if crash_state:
            _event(w, p, "Warning", "BackOff", "Back-off restarting failed container",
                   "1m", count=rng.randint(5, 30))
    if crash_state:
        dep.ready_replicas = 0
    else:
        dep.ready_replicas = max(1, dep.replicas - 1)
    _set_logs(w, pods, [])
    gt = _gt("oom_killed", pods, [["oom", "out of memory", "memory limit"]],
             ["OOMKilled", "Exit Code: 137"], [*BASE_TOOLS, "get_metrics"])
    return w.state(gt)


def gen_image_pull_backoff(rng: random.Random, difficulty: str, pools: NamePools) -> ClusterState:
    w, _ns, app, dep, pods = _setup(rng, difficulty, pools)
    good_tag = dep.image.rsplit(":", 1)[1]
    bad_tag = good_tag + "0" if rng.random() < 0.5 else "v" + good_tag
    bad_image = f"{pools.registry}/{app}:{bad_tag}"
    dep.image, dep.ready_replicas = bad_image, 0
    msg = (f'Failed to pull image "{bad_image}": rpc error: code = NotFound desc = manifest for '
           f"{bad_image} not found: manifest unknown")
    for p in pods:
        p.image, p.phase, p.ready, p.restarts, p.age = bad_image, "ImagePullBackOff", False, 0, "6m"
        p.current_state = ContainerState(
            state="waiting", reason=rng.choice(["ImagePullBackOff", "ImagePullBackOff", "ErrImagePull"]))
        w.logs[f"{p.namespace}/{p.name}"] = []
        w.metrics.pop(f"{p.namespace}/{p.name}", None)
        _event(w, p, "Warning", "Failed", msg, "4m", count=rng.randint(2, 4))
        _event(w, p, "Warning", "Failed", "Error: ErrImagePull", "4m", count=rng.randint(2, 4))
        _event(w, p, "Warning", "BackOff", f'Back-off pulling image "{bad_image}"', "2m",
               count=rng.randint(5, 20))
    gt = _gt("image_pull_backoff", pods, [["image", "tag"], ["pull", "not found", "does not exist"]],
             [f'Failed to pull image "{bad_image}"'], [*BASE_TOOLS, "get_events"])
    return w.state(gt)


def gen_readiness_probe_misconfig(rng: random.Random, difficulty: str, pools: NamePools) -> ClusterState:
    w, _ns, _app, dep, pods = _setup(rng, difficulty, pools)
    app_port = int(pods[0].env["PORT"])
    by_port = rng.random() < 0.5
    if by_port:
        bad_port = 8080 if app_port != 8080 else 8081
        key_lines = [f"INFO listening on :{app_port}"]
        evidence = ["Readiness probe failed", "connection refused", f"listening on :{app_port}"]
    else:
        real_path = rng.choice(["/healthz", "/ready", "/status"])
        bad_path = rng.choice([p for p in ["/health", "/readyz", "/ping"] if p != real_path])
        key_lines = [f"INFO listening on :{app_port}",
                     f"INFO health endpoint registered at {real_path}"]
        evidence = ["Readiness probe failed", "statuscode: 404",
                    f"health endpoint registered at {real_path}"]
    for p in pods:
        p.ready, p.restarts, p.age = False, 0, "8m"
        if by_port:
            p.probes = [Probe(kind="readiness", path="/healthz", port=bad_port, passing=False)]
            message = (f'Readiness probe failed: Get "http://{pod_ip(p)}:{bad_port}/healthz": '
                       f"dial tcp {pod_ip(p)}:{bad_port}: connect: connection refused")
        else:
            p.probes = [Probe(kind="readiness", path=bad_path, port=app_port, passing=False)]
            message = "Readiness probe failed: HTTP probe failed with statuscode: 404"
        _event(w, p, "Warning", "Unhealthy", message, f"{rng.randint(1, 3)}m",
               count=rng.randint(10, 80))
    dep.ready_replicas = 0
    _set_logs(w, pods, key_lines)
    gt = _gt("readiness_probe_misconfig", pods, [["readiness", "probe"], ["port", "path"]],
             evidence, [*BASE_TOOLS, "get_events", "get_logs"])
    return w.state(gt)


def gen_service_selector_mismatch(rng: random.Random, difficulty: str, pools: NamePools) -> ClusterState:
    w, ns, app, _dep, pods = _setup(rng, difficulty, pools)
    svc = next(s for s in w.services if s.namespace == ns and s.name == app)
    if rng.random() < 0.5:
        sel = {"app": app.rsplit("-", 1)[0]}
    else:
        sel = {"name": app}
    svc.selector = sel
    sel_text = ", ".join(f"{k}={v}" for k, v in sorted(sel.items()))
    _set_logs(w, pods, [])
    gt = _gt("service_selector_mismatch", pods, [["selector", "label"], ["service", "endpoint"]],
             ["Endpoints: <none>", f"Selector: {sel_text}"], ["list_pods", "check_service", "describe_pod"],
             obj=f"service/{app}")
    return w.state(gt)


def gen_pending_insufficient_memory(rng: random.Random, difficulty: str, pools: NamePools) -> ClusterState:
    w, _ns, _app, dep, pods = _setup(rng, difficulty, pools)
    for n in w.nodes:
        n.allocated_mem = f"{rng.randint(6, 7)}Gi"
    req = rng.randint(4, 6)
    msg = ("0/3 nodes are available: 3 Insufficient memory. preemption: 0/3 nodes are available: "
           "3 No preemption victims found for incoming pod.")
    for p in pods:
        p.phase, p.ready, p.restarts, p.node, p.age = "Pending", False, 0, None, "9m"
        p.resources.mem_request, p.resources.mem_limit = f"{req}Gi", f"{req + 1}Gi"
        p.current_state = ContainerState(state="waiting", reason="Unschedulable")
        w.logs[f"{p.namespace}/{p.name}"] = []
        w.metrics.pop(f"{p.namespace}/{p.name}", None)
        _event(w, p, "Warning", "FailedScheduling", msg, "9m", count=rng.randint(3, 12))
    dep.ready_replicas = 0
    gt = _gt("pending_insufficient_memory", pods, [["insufficient", "not enough"], ["memory", "resources"]],
             ["0/3 nodes are available: 3 Insufficient memory"], [*BASE_TOOLS, "get_events"])
    return w.state(gt)


def gen_node_memory_pressure_eviction(rng: random.Random, difficulty: str, pools: NamePools) -> ClusterState:
    w, ns, app, dep, pods = _setup(rng, difficulty, pools)
    node = w.nodes[1]
    node.memory_pressure, node.allocated_mem = True, "7Gi"
    keep_running = difficulty != "easy" and len(pods) >= 3
    evicted = pods[:-1] if keep_running else pods
    for p in evicted:
        p.phase, p.ready, p.restarts, p.node = "Evicted", False, 0, node.name
        p.current_state = ContainerState(
            state="terminated", reason="Evicted", message="The node was low on resource: memory.")
        w.logs[f"{p.namespace}/{p.name}"] = []
        w.metrics.pop(f"{p.namespace}/{p.name}", None)
        _event(w, p, "Warning", "Evicted",
               f"The node was low on resource: memory. Container {app} was using "
               f"{rng.randint(900, 1900)}Mi, request is {p.resources.mem_request}, "
               "has larger consumption of memory.", f"{rng.randint(2, 15)}m")
    w.events.append(Event(
        type="Warning", reason="NodeHasMemoryPressure", object_kind="Node", object_name=node.name,
        namespace=ns, message=f"Node {node.name} status is now: NodeHasMemoryPressure", age="12m"))
    dep.ready_replicas = len(pods) - len(evicted)
    gt = _gt("node_memory_pressure_eviction", evicted, [["node"], ["memory pressure", "low on resource", "evict"]],
             ["The node was low on resource: memory"], [*BASE_TOOLS, "get_events"])
    return w.state(gt)


def gen_bad_configmap_value(rng: random.Random, difficulty: str, pools: NamePools) -> ClusterState:
    w, ns, app, dep, pods = _setup(rng, difficulty, pools)
    cm_name = f"{app}-config"
    if rng.random() < 0.5:
        key, val, expect = rng.choice(pools.int_config_keys), rng.choice(BAD_INT_VALUES), "integer"
    else:
        key, val, expect = rng.choice(pools.url_config_keys), rng.choice(BAD_URL_VALUES), "valid URL"
    data = {key: val, "LOG_FORMAT": "json"}
    for k in pools.int_config_keys:
        if k != key and len(data) < 4:
            data[k] = str(rng.randint(5, 50))
    w.configmaps.append(ConfigMap(name=cm_name, namespace=ns, data=data))
    dep.configmap = cm_name
    for p in pods:
        p.env_from_configmap = cm_name
    _crash(w, pods, dep)
    line = f'ERROR config: invalid value for {key}: "{val}" (expected {expect})'
    _set_logs(w, pods, [line, "FATAL: aborting startup"])
    gt = _gt("bad_configmap_value", pods, [["configmap", "config"], [key]],
             [f'invalid value for {key}: "{val}"', f"{key}: {val}"],
             [*BASE_TOOLS, "get_logs", "get_configmap"])
    return w.state(gt)


def gen_hung_process_restart(rng: random.Random, difficulty: str, pools: NamePools) -> ClusterState:
    w, ns, app, dep, pods = _setup(rng, difficulty, pools)
    workers = rng.choice([8, 16, 32])
    key_line = f"WARN worker pool deadlock detected, 0/{workers} workers alive"
    port = int(pods[0].env["PORT"])
    for p in pods:
        p.ready, p.restarts, p.age = False, rng.randint(0, 1), "3d"
        if p.restarts:
            p.last_state = ContainerState(state="terminated", reason="Error", exit_code=1)
        p.probes = [Probe(kind="readiness", path="/healthz", port=port, passing=False)]
        w.metrics[f"{p.namespace}/{p.name}"] = PodMetrics(
            cpu_millicores=rng.randint(0, 2), mem_bytes=parse_mem(p.resources.mem_limit) // 3)
        _event(w, p, "Warning", "Unhealthy",
               f'Readiness probe failed: Get "http://{pod_ip(p)}:{port}/healthz": context deadline '
               "exceeded (Client.Timeout exceeded while awaiting headers)",
               f"{rng.randint(1, 4)}m", count=rng.randint(10, 80))
    dep.ready_replicas = 0
    _set_logs(w, pods, [key_line],
              after_fn=lambda r, a, n: ["WARN readiness check timed out after 5s"] * n)
    fix = Fix(tool="restart_deployment", arguments={"namespace": ns, "deployment": app})
    gt = _gt("hung_process_restart", pods, [["deadlock", "hung", "stuck"]], [key_line],
             [*BASE_TOOLS, "get_logs", "get_metrics"], fix=fix)
    return w.state(gt)


FAILURE_TYPES: dict[str, Callable[[random.Random, str, NamePools], ClusterState]] = {
    "crashloop_missing_env": gen_crashloop_missing_env,
    "oom_killed": gen_oom_killed,
    "image_pull_backoff": gen_image_pull_backoff,
    "readiness_probe_misconfig": gen_readiness_probe_misconfig,
    "service_selector_mismatch": gen_service_selector_mismatch,
    "pending_insufficient_memory": gen_pending_insufficient_memory,
    "node_memory_pressure_eviction": gen_node_memory_pressure_eviction,
    "bad_configmap_value": gen_bad_configmap_value,
    "hung_process_restart": gen_hung_process_restart,
}
FAILURE_TYPE_NAMES = list(FAILURE_TYPES)
# types whose key evidence includes a log line (hard scenarios bury it)
LOG_KEY_TYPES = {
    "crashloop_missing_env", "readiness_probe_misconfig", "bad_configmap_value",
    "hung_process_restart",
}
