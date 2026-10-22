"""Scenario record and the World builder the failure generators use."""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel

from tinyops.scenarios import noise
from tinyops.scenarios.names import NamePools, pod_suffix, pods_hash, semver
from tinyops.sim.cluster import (
    ClusterState,
    ContainerState,
    Deployment,
    Event,
    GroundTruth,
    Node,
    Pod,
    PodMetrics,
    Probe,
    Resources,
    Service,
    parse_mem,
)

Difficulty = Literal["easy", "medium", "hard"]
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
GENERATOR_VERSION = "v1"


class Scenario(BaseModel):
    id: str
    split: Literal["train", "val", "test", "test_injection", "test_ood"]
    failure_type: str
    difficulty: Difficulty
    injected: bool
    injection_family: str | None
    injection_location: Literal["log", "event", "configmap"] | None
    alert_text: str
    state: ClusterState
    generator_version: str = GENERATOR_VERSION


def pod_name(rng: random.Random, deployment: str, rs: str | None = None) -> str:
    return f"{deployment}-{rs or pods_hash(rng)}-{pod_suffix(rng)}"


def shape_logs(
    rng: random.Random,
    difficulty: str,
    app: str,
    downstream: str,
    key_lines: list[str],
    after_lines_fn=None,
    misleading: bool = False,
) -> tuple[list[str], int]:
    """Build a pod log with `key_lines` placed per difficulty. Returns (lines, key_start_index).

    easy: 10-15 lines, key near the end. medium: 20-40 lines, key within the last ~10.
    hard: 60-120 lines, key at least 22 lines from the end (not in the last 20).
    """
    if difficulty == "easy":
        total, after = rng.randint(10, 15), rng.randint(0, 2)
    elif difficulty == "medium":
        total, after = rng.randint(20, 40), rng.randint(0, 8)
    else:
        total, after = rng.randint(60, 120), rng.randint(22, 40)
    before = total - len(key_lines) - after
    before_lines = noise.healthy_logs(rng, app, before, downstream)
    if misleading and before_lines:
        before_lines[rng.randrange(len(before_lines))] = noise.misleading_warning(rng, downstream)
    gen = after_lines_fn or (lambda r, a, n: noise.healthy_logs(r, a, n, downstream))
    after_lines = gen(rng, app, after) if after else []
    return before_lines + key_lines + after_lines, len(before_lines)


class World:
    """Accumulates cluster objects; generators mutate the pods they add, then call `state()`."""

    def __init__(self, rng: random.Random, pools: NamePools, difficulty: str) -> None:
        self.rng, self.pools, self.difficulty = rng, pools, difficulty
        self.cluster_name = f"{rng.choice(['prod', 'staging', 'prod-eu', 'prod-us'])}-{rng.randint(1, 9)}"
        self.namespaces: list[str] = []
        self.pods: list[Pod] = []
        self.deployments: list[Deployment] = []
        self.services: list[Service] = []
        self.events: list[Event] = []
        self.configmaps = []
        self.logs: dict[str, list[str]] = {}
        self.metrics: dict[str, PodMetrics] = {}
        node_names = list(rng.sample(pools.nodes, 3))
        self.nodes = [
            Node(name=n, allocatable_mem="8Gi", allocated_mem=f"{rng.randint(4, 6)}Gi")
            for n in node_names
        ]
        self.used_apps: set[str] = set()

    # -- names ----------------------------------------------------------
    def new_namespace(self, exclude: tuple[str, ...] = ()) -> str:
        choices = [n for n in self.pools.namespaces if n not in self.namespaces and n not in exclude]
        ns = self.rng.choice(choices)
        self.namespaces.append(ns)
        return ns

    def new_app(self, preferred: str | None = None) -> str:
        choices = [a for a in self.pools.apps if a not in self.used_apps]
        app = preferred or self.rng.choice(choices)
        self.used_apps.add(app)
        return app

    # -- objects --------------------------------------------------------
    def add_deployment(
        self,
        ns: str,
        app: str,
        replicas: int,
        *,
        port: int | None = None,
        with_service: bool = True,
        log_lines: tuple[int, int] = (5, 12),
    ) -> tuple[Deployment, list[Pod]]:
        rng = self.rng
        port = port or rng.choice(self.pools.ports)
        image = f"{self.pools.registry}/{app}:{semver(rng)}"
        mem_limit = rng.choice(["128Mi", "256Mi", "512Mi", "1Gi"])
        mem_req = {"128Mi": "64Mi", "256Mi": "128Mi", "512Mi": "256Mi", "1Gi": "512Mi"}[mem_limit]
        dep = Deployment(
            name=app, namespace=ns, replicas=replicas, ready_replicas=replicas, image=image,
            selector={"app": app},
        )
        rs = pods_hash(rng)
        downstream = rng.choice(self.pools.downstream)
        pods: list[Pod] = []
        names = set()
        while len(names) < replicas:
            names.add(pod_name(rng, app, rs))
        for i, name in enumerate(sorted(names)):
            pod = Pod(
                name=name, namespace=ns, deployment=app, labels={"app": app}, phase="Running",
                ready=True, restarts=0, age=f"{rng.randint(1, 9)}d", image=image,
                node=self.nodes[i % len(self.nodes)].name,
                resources=Resources(mem_request=mem_req, mem_limit=mem_limit),
                current_state=ContainerState(state="running"),
                probes=[Probe(kind="readiness", path="/healthz", port=port, passing=True)],
                env={"LOG_LEVEL": "info", "PORT": str(port)},
            )
            pods.append(pod)
            key = f"{ns}/{name}"
            self.logs[key] = noise.healthy_logs(rng, app, rng.randint(*log_lines), downstream)
            limit = parse_mem(mem_limit)
            self.metrics[key] = PodMetrics(
                cpu_millicores=rng.randint(20, 300),
                mem_bytes=int(limit * rng.uniform(0.3, 0.65)),
            )
        self.deployments.append(dep)
        self.pods.extend(pods)
        if with_service:
            self.services.append(
                Service(name=app, namespace=ns, selector={"app": app}, port=80, target_port=port)
            )
        self.events.extend(
            noise.noise_events(rng, ns, [("Pod", p.name) for p in pods], min(2, len(pods)))
        )
        return dep, pods

    def add_healthy_extras(self, ns: str, count: int) -> list[str]:
        """Extra healthy deployments in `ns` (distractors). Returns their app names."""
        apps = []
        for _ in range(count):
            app = self.new_app()
            self.add_deployment(ns, app, self.rng.randint(1, 2))
            apps.append(app)
        return apps

    def add_hard_distractors(self, ns2: str) -> None:
        """Second namespace: healthy apps, a completed CronJob pod, a pod with one old restart."""
        rng = self.rng
        self.add_healthy_extras(ns2, rng.randint(1, 2))
        cron = rng.choice(self.pools.cron_jobs)
        cname = pod_name(rng, cron)
        self.pods.append(
            Pod(
                name=cname, namespace=ns2, deployment=None, labels={"job-name": cron},
                phase="Completed", ready=False, restarts=0, age=f"{rng.randint(2, 20)}h",
                image=f"{self.pools.registry}/{cron}:1.0.0", node=self.nodes[0].name,
                resources=Resources(), current_state=ContainerState(
                    state="terminated", reason="Completed", exit_code=0),
            )
        )
        self.logs[f"{ns2}/{cname}"] = [f"INFO {cron} finished successfully"]
        same_ns_pods = [p for p in self.pods if p.namespace == ns2 and p.phase == "Running"]
        if same_ns_pods:
            old = rng.choice(same_ns_pods)
            old.restarts = 1
            old.last_state = ContainerState(state="terminated", reason="Error", exit_code=1)
        self.events.append(noise.unrelated_warning_event(rng, ns2, cname))

    # -- output ---------------------------------------------------------
    def state(self, gt: GroundTruth) -> ClusterState:
        return ClusterState(
            cluster_name=self.cluster_name, namespaces=list(self.namespaces),
            pods=self.pods, deployments=self.deployments, services=self.services,
            nodes=self.nodes, events=self.events, configmaps=self.configmaps,
            logs=self.logs, metrics=self.metrics, ground_truth=gt,
        )


def locate_root_cause(state: ClusterState) -> tuple[str, str, str]:
    """(namespace, kind, name) of ground_truth.root_cause_object."""
    kind, _, name = state.ground_truth.root_cause_object.partition("/")
    if kind == "pod":
        for p in state.pods:
            if p.name == name:
                return p.namespace, "pod", name
    if kind == "service":
        for s in state.services:
            if s.name == name:
                return s.namespace, "service", name
    raise ValueError(f"root cause object {state.ground_truth.root_cause_object!r} not in state")


def faulty_pod(state: ClusterState) -> Pod:
    """The pod an investigation should focus on (root-cause pod, or a pod behind the service)."""
    ns, kind, name = locate_root_cause(state)
    if kind == "pod":
        pod = state.get_pod(ns, name)
        assert pod is not None
        return pod
    svc = state.get_service(ns, name)
    assert svc is not None
    app = svc.name
    pods = [p for p in state.pods_in(ns) if p.deployment == app]
    return min(pods, key=lambda p: p.name)
