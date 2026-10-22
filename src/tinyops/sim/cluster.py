"""Pydantic state for the fake Kubernetes cluster that KubeSim simulates."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

PodPhase = Literal[
    "Running", "Pending", "CrashLoopBackOff", "ImagePullBackOff", "Error", "Completed", "Evicted"
]

_MEM_UNITS = {
    "": 1,
    "k": 1000, "m": 1000**2, "g": 1000**3, "t": 1000**4,
    "ki": 1024, "mi": 1024**2, "gi": 1024**3, "ti": 1024**4,
}
_MEM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*$")


class _Model(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")


def parse_mem(s: str) -> int:
    """Parse a Kubernetes memory quantity ("256Mi", "1Gi", "500M", "1024") to bytes."""
    m = _MEM_RE.match(s)
    if not m or m.group(2).lower() not in _MEM_UNITS:
        raise ValueError(f"cannot parse memory quantity {s!r}")
    return int(float(m.group(1)) * _MEM_UNITS[m.group(2).lower()])


class ContainerState(_Model):
    state: Literal["running", "waiting", "terminated"]
    reason: str | None = None
    exit_code: int | None = None
    message: str | None = None


class Probe(_Model):
    kind: Literal["readiness", "liveness"]
    path: str
    port: int
    passing: bool


class Resources(_Model):
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    mem_request: str = "128Mi"
    mem_limit: str = "256Mi"


class Pod(_Model):
    name: str
    namespace: str
    deployment: str | None
    labels: dict[str, str]
    phase: PodPhase
    ready: bool
    restarts: int
    age: str
    image: str
    node: str | None
    resources: Resources
    current_state: ContainerState
    last_state: ContainerState | None = None
    probes: list[Probe] = []
    env: dict[str, str] = {}
    env_from_configmap: str | None = None


class Deployment(_Model):
    name: str
    namespace: str
    replicas: int
    ready_replicas: int
    image: str
    selector: dict[str, str]
    configmap: str | None = None


class Service(_Model):
    name: str
    namespace: str
    selector: dict[str, str]
    port: int
    target_port: int


class Node(_Model):
    name: str
    ready: bool = True
    memory_pressure: bool = False
    disk_pressure: bool = False
    allocatable_mem: str = "8Gi"
    allocated_mem: str = "5Gi"


class Event(_Model):
    type: Literal["Normal", "Warning"]
    reason: str
    object_kind: str
    object_name: str
    namespace: str
    message: str
    count: int = 1
    age: str = "1m"


class ConfigMap(_Model):
    name: str
    namespace: str
    data: dict[str, str]


class PodMetrics(_Model):
    cpu_millicores: int
    mem_bytes: int


class Fix(_Model):
    tool: str
    arguments: dict[str, Any]


class GroundTruth(_Model):
    failure_type: str
    root_cause_object: str
    # AND of ORs, e.g. [["env", "environment variable"], ["DATABASE_URL"]]
    root_cause_keywords: list[list[str]]
    key_evidence: list[str]  # exact substrings that appear in some tool output
    relevant_tools: list[str]
    correct_fix: Fix | None = None
    injected_payload: str | None = None
    injection_targets: list[str] = []


_AGE_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def age_seconds(age: str) -> int:
    """Convert "5m" / "2d" / "30s" to seconds (used to order events newest first)."""
    m = re.match(r"^(\d+)([smhd])$", age.strip())
    if not m:
        raise ValueError(f"cannot parse age {age!r}")
    return int(m.group(1)) * _AGE_UNITS[m.group(2)]


class ClusterState(_Model):
    cluster_name: str
    namespaces: list[str]
    pods: list[Pod]
    deployments: list[Deployment]
    services: list[Service]
    nodes: list[Node]
    events: list[Event]
    configmaps: list[ConfigMap]
    logs: dict[str, list[str]]  # key = f"{namespace}/{pod}"
    metrics: dict[str, PodMetrics]  # key = f"{namespace}/{pod}"
    ground_truth: GroundTruth  # NEVER shown to the agent

    def get_pod(self, ns: str, name: str) -> Pod | None:
        return next((p for p in self.pods if p.namespace == ns and p.name == name), None)

    def pods_in(self, ns: str) -> list[Pod]:
        return [p for p in self.pods if p.namespace == ns]

    def get_service(self, ns: str, name: str) -> Service | None:
        return next((s for s in self.services if s.namespace == ns and s.name == name), None)

    def get_deployment(self, ns: str, name: str) -> Deployment | None:
        return next((d for d in self.deployments if d.namespace == ns and d.name == name), None)

    def get_configmap(self, ns: str, name: str) -> ConfigMap | None:
        return next((c for c in self.configmaps if c.namespace == ns and c.name == name), None)

    def events_for(self, ns: str, object_name: str | None = None) -> list[Event]:
        return [
            e
            for e in self.events
            if e.namespace == ns and (not object_name or e.object_name == object_name)
        ]

    def pods_matching(self, ns: str, selector: dict[str, str]) -> list[Pod]:
        if not selector:
            return []
        return [
            p
            for p in self.pods_in(ns)
            if all(p.labels.get(k) == v for k, v in selector.items())
        ]
