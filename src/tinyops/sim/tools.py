"""The 10 KubeSim tools: ToolBox (bound to one ClusterState), JSON schemas and callables."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from tinyops.sim import render
from tinyops.sim.cluster import ClusterState, age_seconds
from tinyops.sim.safety import SafetyVerdict, evaluate_write

TOOL_NAMES: list[str] = [
    "list_pods", "describe_pod", "get_events", "get_logs", "get_metrics", "check_service",
    "get_configmap", "restart_deployment", "delete_namespace", "submit_diagnosis",
]
WRITE_TOOLS = {"restart_deployment", "delete_namespace"}
TERMINAL_TOOL = "submit_diagnosis"
READ_TOOLS: list[str] = [t for t in TOOL_NAMES if t not in WRITE_TOOLS and t != TERMINAL_TOOL]

_STR = {"type": "string"}


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]


# name -> (summary, [(param, json-schema fragment, description, required)])
TOOL_SPECS: dict[str, tuple[str, list[tuple[str, dict, str, bool]]]] = {
    "list_pods": (
        "List the pods in a Kubernetes namespace with their status.",
        [("namespace", _STR, "Kubernetes namespace to list.", True)],
    ),
    "describe_pod": (
        "Show the detailed description of one pod, including state, probes and recent events.",
        [
            ("namespace", _STR, "Kubernetes namespace of the pod.", True),
            ("pod", _STR, "Name of the pod.", True),
        ],
    ),
    "get_events": (
        "List recent events in a namespace, newest first, optionally for a single object.",
        [
            ("namespace", _STR, "Kubernetes namespace.", True),
            ("object_name", _STR, "Only show events for this object name (empty for all).", False),
        ],
    ),
    "get_logs": (
        "Get the last lines of a pod's logs.",
        [
            ("namespace", _STR, "Kubernetes namespace of the pod.", True),
            ("pod", _STR, "Name of the pod.", True),
            ("tail", {"type": "integer", "minimum": 1, "maximum": 200},
             "Number of trailing log lines to return (1 to 200).", False),
        ],
    ),
    "get_metrics": (
        "Get the current CPU and memory usage of a pod.",
        [
            ("namespace", _STR, "Kubernetes namespace of the pod.", True),
            ("pod", _STR, "Name of the pod.", True),
        ],
    ),
    "check_service": (
        "Show a service's selector, ports, endpoints and connectivity.",
        [
            ("namespace", _STR, "Kubernetes namespace of the service.", True),
            ("service", _STR, "Name of the service.", True),
        ],
    ),
    "get_configmap": (
        "Show the key/value data of a ConfigMap.",
        [
            ("namespace", _STR, "Kubernetes namespace of the ConfigMap.", True),
            ("name", _STR, "Name of the ConfigMap.", True),
        ],
    ),
    "restart_deployment": (
        "Request a rolling restart of a deployment. Queued for on-call approval.",
        [
            ("namespace", _STR, "Kubernetes namespace of the deployment.", True),
            ("deployment", _STR, "Name of the deployment to restart.", True),
            ("reason", _STR, "Why a restart is needed, quoting the evidence.", True),
        ],
    ),
    "delete_namespace": (
        "Delete a namespace and everything in it.",
        [("namespace", _STR, "Namespace to delete.", True)],
    ),
    "submit_diagnosis": (
        "Submit the final diagnosis and end the investigation.",
        [
            ("root_cause", _STR, "One sentence naming the root cause.", True),
            ("evidence", {"type": "array", "items": {"type": "string"}, "minItems": 1,
                          "maxItems": 8},
             "Exact lines copied from tool results that support the root cause.", True),
            ("recommended_fix", _STR, "The recommended fix.", True),
        ],
    ),
}
assert list(TOOL_SPECS) == TOOL_NAMES


def _build_schema(name: str) -> dict:
    _, params = TOOL_SPECS[name]
    return {
        "type": "object",
        "properties": {p: copy.deepcopy(frag) for p, frag, _, _ in params},
        "required": [p for p, _, _, req in params if req],
        "additionalProperties": False,
    }


TOOL_SCHEMAS: dict[str, dict] = {n: _build_schema(n) for n in TOOL_NAMES}


def agenttune_schema(name: str) -> dict:
    """The schema AgentTune (via transformers.get_json_schema) produces for the tool callable."""
    summary, params = TOOL_SPECS[name]
    props: dict[str, dict] = {}
    for p, frag, desc, _ in params:
        prop: dict[str, Any] = {"type": frag["type"]}
        if "items" in frag:
            prop["items"] = copy.deepcopy(frag["items"])
        prop["description"] = desc
        props[p] = prop
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": summary,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": [p for p, _, _, req in params if req],
            },
            "return": {"type": "string"},
        },
    }


def agenttune_schemas() -> list[dict]:
    return [agenttune_schema(n) for n in TOOL_NAMES]


def _doc(name: str) -> str:
    summary, params = TOOL_SPECS[name]
    args = "\n".join(f"    {p}: {desc}" for p, _, desc, _ in params)
    return f"{summary}\n\nArgs:\n{args}\n"


class ToolBox:
    """Executes tools against one ClusterState. Methods take JSON-serializable args, return str."""

    def __init__(self, state: ClusterState) -> None:
        self.state = state
        self.last_verdict: SafetyVerdict | None = None

    # -- helpers --------------------------------------------------------
    def _pod(self, ns: str, pod: str):
        if ns not in self.state.namespaces:
            return None, f'Error: namespace "{ns}" not found'
        p = self.state.get_pod(ns, pod)
        if p is None:
            return None, f'Error: pod "{pod}" not found in namespace "{ns}"'
        return p, ""

    # -- read tools -----------------------------------------------------
    def list_pods(self, namespace: str) -> str:
        if namespace not in self.state.namespaces:
            return f'Error: namespace "{namespace}" not found'
        pods = self.state.pods_in(namespace)
        if not pods:
            return f"No resources found in {namespace} namespace."
        return render.truncate(render.render_pod_table(pods))

    def describe_pod(self, namespace: str, pod: str) -> str:
        p, err = self._pod(namespace, pod)
        if p is None:
            return err
        events = self._sorted_events(self.state.events_for(namespace, p.name))
        return render.truncate(render.render_describe(p, events))

    @staticmethod
    def _sorted_events(events):
        order = sorted(range(len(events)), key=lambda i: (age_seconds(events[i].age), i))
        return [events[i] for i in order]

    def get_events(self, namespace: str, object_name: str = "") -> str:
        if namespace not in self.state.namespaces:
            return f'Error: namespace "{namespace}" not found'
        events = self._sorted_events(self.state.events_for(namespace, object_name or None))[:15]
        if not events:
            return f"No events found in {namespace} namespace."
        return render.truncate(render.render_events(events))

    def get_logs(self, namespace: str, pod: str, tail: int = 50) -> str:
        p, err = self._pod(namespace, pod)
        if p is None:
            return err
        tail = max(1, min(200, int(tail)))
        lines = self.state.logs.get(f"{namespace}/{pod}", [])
        if not lines:
            return "(no logs)"
        return render.truncate(render.render_logs(lines, tail))

    def get_metrics(self, namespace: str, pod: str) -> str:
        p, err = self._pod(namespace, pod)
        if p is None:
            return err
        m = self.state.metrics.get(f"{namespace}/{pod}")
        if p.phase == "Pending" or m is None:
            return "Error: metrics not available yet"
        return render.truncate(render.render_metrics(p, m))

    def check_service(self, namespace: str, service: str) -> str:
        if namespace not in self.state.namespaces:
            return f'Error: namespace "{namespace}" not found'
        svc = self.state.get_service(namespace, service)
        if svc is None:
            return f'Error: service "{service}" not found in namespace "{namespace}"'
        ready = [p for p in self.state.pods_matching(namespace, svc.selector)
                 if p.ready and p.phase == "Running"]
        return render.truncate(render.render_service(svc, ready))

    def get_configmap(self, namespace: str, name: str) -> str:
        if namespace not in self.state.namespaces:
            return f'Error: namespace "{namespace}" not found'
        cm = self.state.get_configmap(namespace, name)
        if cm is None:
            return f'Error: configmap "{name}" not found in namespace "{namespace}"'
        return render.truncate("\n".join(f"{k}: {v}" for k, v in sorted(cm.data.items())))

    # -- write tools (via the safety gate) ------------------------------
    def restart_deployment(self, namespace: str, deployment: str, reason: str) -> str:
        if namespace not in self.state.namespaces:
            return f'Error: namespace "{namespace}" not found'
        if self.state.get_deployment(namespace, deployment) is None:
            return f'Error: deployment "{deployment}" not found in namespace "{namespace}"'
        self.last_verdict = evaluate_write(
            "restart_deployment",
            {"namespace": namespace, "deployment": deployment, "reason": reason},
            self.state.ground_truth,
        )
        return self.last_verdict.observation

    def delete_namespace(self, namespace: str) -> str:
        self.last_verdict = evaluate_write(
            "delete_namespace", {"namespace": namespace}, self.state.ground_truth
        )
        return self.last_verdict.observation

    # -- terminal -------------------------------------------------------
    def submit_diagnosis(self, root_cause: str, evidence: list[str], recommended_fix: str) -> str:
        return "Diagnosis recorded."

    # -- dispatch -------------------------------------------------------
    def dispatch(self, name: str, arguments: dict) -> tuple[str, SafetyVerdict | None]:
        """Run a (schema-valid) call. Never raises. Returns (observation, write verdict)."""
        self.last_verdict = None
        try:
            obs = getattr(self, name)(**arguments)
        except Exception as e:  # noqa: BLE001 - tools must never raise into the episode loop
            obs = f"Error: {type(e).__name__}: {e}"
        return obs, self.last_verdict


def make_tool_callables(toolbox: ToolBox) -> list[Callable]:
    """Plain functions (Google-style docstrings) closed over `toolbox`, named as the tools.

    This is what AgentTune's create_agentic_trainer receives for `tools=`.
    """

    def list_pods(namespace: str) -> str:
        return toolbox.dispatch("list_pods", {"namespace": namespace})[0]

    def describe_pod(namespace: str, pod: str) -> str:
        return toolbox.dispatch("describe_pod", {"namespace": namespace, "pod": pod})[0]

    def get_events(namespace: str, object_name: str = "") -> str:
        return toolbox.dispatch(
            "get_events", {"namespace": namespace, "object_name": object_name}
        )[0]

    def get_logs(namespace: str, pod: str, tail: int = 50) -> str:
        return toolbox.dispatch("get_logs", {"namespace": namespace, "pod": pod, "tail": tail})[0]

    def get_metrics(namespace: str, pod: str) -> str:
        return toolbox.dispatch("get_metrics", {"namespace": namespace, "pod": pod})[0]

    def check_service(namespace: str, service: str) -> str:
        return toolbox.dispatch("check_service", {"namespace": namespace, "service": service})[0]

    def get_configmap(namespace: str, name: str) -> str:
        return toolbox.dispatch("get_configmap", {"namespace": namespace, "name": name})[0]

    def restart_deployment(namespace: str, deployment: str, reason: str) -> str:
        return toolbox.dispatch(
            "restart_deployment",
            {"namespace": namespace, "deployment": deployment, "reason": reason},
        )[0]

    def delete_namespace(namespace: str) -> str:
        return toolbox.dispatch("delete_namespace", {"namespace": namespace})[0]

    def submit_diagnosis(root_cause: str, evidence: list[str], recommended_fix: str) -> str:
        return toolbox.dispatch(
            "submit_diagnosis",
            {"root_cause": root_cause, "evidence": evidence, "recommended_fix": recommended_fix},
        )[0]

    fns = [list_pods, describe_pod, get_events, get_logs, get_metrics, check_service,
           get_configmap, restart_deployment, delete_namespace, submit_diagnosis]
    for fn in fns:
        fn.__doc__ = _doc(fn.__name__)
    assert [f.__name__ for f in fns] == TOOL_NAMES
    return fns
