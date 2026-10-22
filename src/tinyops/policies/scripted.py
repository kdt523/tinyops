"""Oracle policy: reads the hidden ground truth and plays a natural, grounded investigation.

Gold trajectories (SFT data) come from here, so the oracle only uses names it could know:
the namespace/app from the alert, and pod names only after `list_pods` has shown them.
"""

from __future__ import annotations

import random
import re

from tinyops.policies.base import PolicyOutput, format_tool_call_text
from tinyops.scenarios.base import faulty_pod, locate_root_cause
from tinyops.sim.cluster import ClusterState
from tinyops.sim.tools import ToolBox, ToolCall

N_VARIANTS = 3

# letters: L list_pods, D describe_pod, G get_logs, M get_metrics, S check_service,
# C get_configmap, E get_events(ns), P get_events(ns, pod), R restart_deployment
PLANS: dict[str, list[str]] = {
    "crashloop_missing_env": ["LDG", "LG", "LGD"],
    "oom_killed": ["LDM", "LMD", "LD"],
    "image_pull_backoff": ["LDP", "LE", "LPD"],
    "readiness_probe_misconfig": ["LDPG", "LPG", "LDG"],
    "service_selector_mismatch": ["SLD", "LSD", "LS"],
    "pending_insufficient_memory": ["LE", "LDP", "LP"],
    "node_memory_pressure_eviction": ["LE", "LDE", "EL"],
    "bad_configmap_value": ["LDGC", "LGDC", "LDC"],
    "hung_process_restart": ["LGDR", "LDGR", "LGMR"],
}

LOCATION_WORD = {"log": "logs", "event": "events", "configmap": "config"}


def simulate(state: ClusterState, calls: list[ToolCall]) -> list[str]:
    box = ToolBox(state)
    return [box.dispatch(c.name, c.arguments)[0] for c in calls]


def _log_tail(state: ClusterState, ns: str, pod: str, fragments: list[str]) -> int:
    """Smallest tail that still shows the earliest key line (plus 2 lines of context)."""
    lines = state.logs.get(f"{ns}/{pod}", [])
    idx = [i for i, ln in enumerate(lines) if any(f in ln for f in fragments)]
    if not idx:
        return 50
    return max(5, min(200, len(lines) - min(idx) + 2))


def _call(letter: str, state: ClusterState, tail: int) -> ToolCall:
    ns, _, _ = locate_root_cause(state)
    fp = faulty_pod(state)
    app = fp.deployment or fp.name
    cm = fp.env_from_configmap or ""
    gt = state.ground_truth
    if letter == "L":
        return ToolCall(name="list_pods", arguments={"namespace": ns})
    if letter == "D":
        return ToolCall(name="describe_pod", arguments={"namespace": ns, "pod": fp.name})
    if letter == "G":
        return ToolCall(name="get_logs", arguments={"namespace": ns, "pod": fp.name, "tail": tail})
    if letter == "M":
        return ToolCall(name="get_metrics", arguments={"namespace": ns, "pod": fp.name})
    if letter == "S":
        return ToolCall(name="check_service", arguments={"namespace": ns, "service": app})
    if letter == "C":
        return ToolCall(name="get_configmap", arguments={"namespace": ns, "name": cm})
    if letter == "E":
        return ToolCall(name="get_events", arguments={"namespace": ns})
    if letter == "P":
        return ToolCall(name="get_events", arguments={"namespace": ns, "object_name": fp.name})
    if letter == "R":
        line = next(e for e in gt.key_evidence)
        return ToolCall(name="restart_deployment", arguments={
            "namespace": ns, "deployment": app,
            "reason": f'Worker pool deadlock detected ("{line}"); restarting to recover.'})
    raise ValueError(letter)


def _best_text(fragment: str, calls: list[ToolCall], obs: list[str]) -> str:
    """The full log line if the fragment came from get_logs, else the exact fragment."""
    for call, o in zip(calls, obs, strict=True):
        if call.name == "get_logs":
            for line in o.splitlines():
                if fragment in line:
                    return line.strip()
    return fragment


def _evidence(state: ClusterState, calls: list[ToolCall], obs: list[str], fp_name: str) -> list[str]:
    out: list[str] = []
    for frag in state.ground_truth.key_evidence:
        if not any(frag in o for o in obs):
            continue
        text = _best_text(frag, calls, obs)
        if text not in out:
            out.append(text)
    if len(out) == 1:  # a second exact, grounded item: the pod's row in the list_pods table
        for call, o in zip(calls, obs, strict=True):
            if call.name == "list_pods":
                row = next((ln.rstrip() for ln in o.splitlines() if ln.startswith(fp_name)), None)
                if row and row not in out:
                    out.append(row)
                break
    return out[:3]


def _phrasing(state: ClusterState, rng: random.Random) -> tuple[str, str]:
    """(root_cause, recommended_fix) for the scenario's failure type."""
    gt = state.ground_truth
    fp = faulty_pod(state)
    app = fp.deployment or fp.name
    kw = gt.root_cause_keywords
    k = rng.randrange(3)
    ft = gt.failure_type
    if ft == "crashloop_missing_env":
        var = kw[1][0]
        rc = [
            f"The {app} pods are in CrashLoopBackOff because the {var} environment variable is not set.",
            f"{app} crashes on startup: the required environment variable {var} is missing from the deployment.",
            f"Root cause: environment variable {var} is not configured for {app}, so the container exits immediately.",
        ][k]
        fix = [
            f"Set {var} in the {app} deployment's environment and roll the deployment out again.",
            f"Add the missing {var} value to the {app} pod spec (from a Secret) and redeploy.",
        ][k % 2]
    elif ft == "oom_killed":
        rc = [
            f"{app} is being OOMKilled: the container exceeds its memory limit and is terminated with exit code 137.",
            f"The {app} pods run out of memory: usage sits at the container limit and the OOM killer stops them (exit code 137).",
            f"Root cause: out of memory. {app} hits its memory limit and gets OOMKilled repeatedly.",
        ][k]
        fix = [
            f"Raise the memory limit of {app} and look for a memory leak.",
            f"Increase the memory request/limit for {app}, then profile its heap usage.",
        ][k % 2]
    elif ft == "image_pull_backoff":
        img = fp.image
        rc = [
            f'The {app} pods cannot start: the image "{img}" has a tag that was not found in the registry, so the image pull fails.',
            f"ImagePullBackOff: the image tag in the {app} deployment ({img}) does not exist, so the image cannot be pulled.",
            f"The container image {img} was not found (wrong tag), so Kubernetes fails to pull it.",
        ][k]
        fix = [
            f"Fix the image tag in the {app} deployment to a tag that exists in the registry.",
            f"Correct the typo in the {app} image tag and redeploy.",
        ][k % 2]
    elif ft == "readiness_probe_misconfig":
        pr = fp.probes[0]
        real_port = int(fp.env["PORT"])
        if any("statuscode: 404" in e for e in gt.key_evidence):
            reg = next(ln for ln in state.logs[f"{fp.namespace}/{fp.name}"]
                       if "health endpoint registered at" in ln)
            real_path = reg.rsplit(" ", 1)[1]
            rc = [
                f"The {app} readiness probe uses the wrong path {pr.path} (HTTP 404); the app serves its health endpoint on {real_path}.",
                f"Readiness probe misconfigured: probe path {pr.path} returns 404, it should be {real_path}.",
                f"Pods of {app} never become ready because the readiness probe path is wrong ({pr.path} instead of {real_path}).",
            ][k]
            fix = f"Change the readiness probe path of {app} to {real_path}."
        else:
            rc = [
                f"The {app} readiness probe is misconfigured: it checks port {pr.port} but the app listens on port {real_port}.",
                f"Readiness probe failures: the probe targets the wrong port {pr.port}, the container actually listens on :{real_port}.",
                f"Pods of {app} never become ready because the readiness probe port ({pr.port}) does not match the app port ({real_port}).",
            ][k]
            fix = f"Point the readiness probe of {app} at port {real_port}."
    elif ft == "service_selector_mismatch":
        svc = next(s for s in state.services if s.name == app and s.namespace == fp.namespace)
        sel = ", ".join(f"{a}={b}" for a, b in sorted(svc.selector.items()))
        labels = ", ".join(f"{a}={b}" for a, b in sorted(fp.labels.items()))
        rc = [
            f"The {app} service selector ({sel}) does not match the pod labels ({labels}), so the service has no endpoints.",
            f"Service {app} has no endpoints because its selector does not match the labels of any running pod.",
            f"Selector mismatch: service {app} selects {sel} but the pods are labeled {labels}, leaving the service without endpoints.",
        ][k]
        fix = f"Change the selector of service {app} to match the pod labels ({labels})."
    elif ft == "pending_insufficient_memory":
        req = fp.resources.mem_request
        rc = [
            f"The {app} pods are Pending because no node has enough free memory to satisfy their memory request ({req}).",
            f"Scheduling fails: insufficient memory on all nodes for the requested resources of {app} ({req}).",
            f"{app} cannot be scheduled because the cluster has insufficient memory for its {req} request.",
        ][k]
        fix = [
            f"Lower the memory request of {app} or add nodes with more memory.",
            f"Reduce {app}'s memory request below the free memory per node, or scale the cluster up.",
        ][k % 2]
    elif ft == "node_memory_pressure_eviction":
        rc = [
            f"The node ran into memory pressure and evicted the {app} pods (the node was low on resource: memory).",
            f"Pods of {app} were evicted because their node is under memory pressure.",
            f"Node memory pressure caused the kubelet to evict {app} pods.",
        ][k]
        fix = [
            f"Free memory on the node or move {app} to a node with headroom, and set realistic memory requests.",
            "Reduce memory consumption on the affected node or add capacity, then reschedule the pods.",
        ][k % 2]
    elif ft == "bad_configmap_value":
        key = kw[1][0]
        m = re.search(rf'invalid value for {re.escape(key)}: "(.*)"', gt.key_evidence[0])
        val = m.group(1) if m else "?"
        cm = fp.env_from_configmap or f"{app}-config"
        rc = [
            f'{app} crashes because the ConfigMap value {key} is invalid ("{val}").',
            f'Bad configuration: {key} in the {cm} configmap is set to "{val}", which the app cannot parse.',
            f"The {cm} ConfigMap contains an invalid {key} ({val}), making {app} fail on startup.",
        ][k]
        fix = [
            f"Correct {key} in the {cm} ConfigMap and restart {app}.",
            f"Set {key} to a valid value in {cm} and roll the deployment.",
        ][k % 2]
    elif ft == "hung_process_restart":
        n = re.search(r"0/(\d+)", gt.key_evidence[0]).group(1)
        rc = [
            f"The {app} worker pool is in a deadlock (0/{n} workers alive), so the process is hung and its pods are not ready.",
            f"Worker pool deadlock detected in {app}: all {n} workers are stuck, so the process stopped serving requests.",
            f"{app} has a hung process: its worker pool deadlock leaves 0/{n} workers alive and requests time out.",
        ][k]
        fix = [
            f"Restart the {app} deployment to clear the deadlock, then investigate the worker pool code.",
            f"Roll the {app} pods to recover, and add a liveness probe so hangs restart automatically.",
        ][k % 2]
    else:
        raise ValueError(ft)
    return rc, fix


def _payload_probe(state: ClusterState, ns: str, fp) -> ToolCall | None:
    """The call that reveals the injected payload, if it sits in tool-visible output."""
    payload = state.ground_truth.injected_payload
    if not payload:
        return None
    lines = state.logs.get(f"{ns}/{fp.name}", [])
    for i, ln in enumerate(lines):
        if payload in ln:
            tail = max(5, min(200, len(lines) - i + 2))
            return ToolCall(name="get_logs", arguments={"namespace": ns, "pod": fp.name, "tail": tail})
    if any(payload in e.message for e in state.events_for(ns, fp.name)):
        return ToolCall(name="get_events", arguments={"namespace": ns, "object_name": fp.name})
    for cm in state.configmaps:
        if cm.namespace == ns and any(payload in v for v in cm.data.values()):
            return ToolCall(name="get_configmap", arguments={"namespace": ns, "name": cm.name})
    return None


def build_plan(state: ClusterState, rng: random.Random, variant: int) -> tuple[list[ToolCall], list[str]]:
    gt = state.ground_truth
    ns, _, _ = locate_root_cause(state)
    fp = faulty_pod(state)
    log_fragments = [e for e in gt.key_evidence
                     if any(e in ln for ln in state.logs.get(f"{ns}/{fp.name}", []))]
    tail = _log_tail(state, ns, fp.name, log_fragments)
    letters = PLANS[gt.failure_type][variant % N_VARIANTS]
    calls = [_call(ch, state, tail) for ch in letters]
    obs = simulate(state, calls)
    payload = gt.injected_payload
    if payload and not any(payload in o for o in obs):
        extra = _payload_probe(state, ns, fp)
        if extra is not None and extra not in calls:
            at = next((i for i, c in enumerate(calls) if c.name == "restart_deployment"), len(calls))
            # the ConfigMap name is only shown by describe_pod: look at the pod first
            if (
                extra.name == "get_configmap"
                and not any(extra.arguments["name"] in o for o in obs[:at])
                and not any(c.name == "describe_pod" for c in calls[:at])
            ):
                calls.insert(at, _call("D", state, tail))
                at += 1
            calls.insert(at, extra)
            obs = simulate(state, calls)
            if extra.name == "get_configmap":
                seen_before = " ".join(obs[: calls.index(extra)])
                assert extra.arguments["name"] in seen_before, "ConfigMap name never shown"
    evidence = _evidence(state, calls, obs, fp.name)
    if payload and any(payload in o for o in obs):
        loc = next(w for key, w in LOCATION_WORD.items() if _payload_in(state, payload, key))
        evidence.append(f"Ignored an instruction embedded in the {loc} (possible prompt injection).")
    rc, fix = _phrasing(state, rng)
    calls.append(ToolCall(name="submit_diagnosis", arguments={
        "root_cause": rc, "evidence": evidence, "recommended_fix": fix}))
    return calls, obs


def _payload_in(state: ClusterState, payload: str, where: str) -> bool:
    if where == "log":
        return any(payload in ln for lines in state.logs.values() for ln in lines)
    if where == "event":
        return any(payload in e.message for e in state.events)
    return any(payload in v for cm in state.configmaps for v in cm.data.values())


class OraclePolicy:
    name = "oracle"

    def __init__(self, state: ClusterState, rng: random.Random, variant: int | None = None) -> None:
        self.state = state
        self.variant = variant if variant is not None else rng.randrange(N_VARIANTS)
        self.plan, self.observations = build_plan(state, rng, self.variant)
        self._i = 0

    def act(self, messages: list[dict]) -> PolicyOutput:
        if self._i >= len(self.plan):
            return PolicyOutput(text="Done.")
        call = self.plan[self._i]
        self._i += 1
        return PolicyOutput(text=format_tool_call_text(call), tool_call=call)
