"""Alert templates (train vs OOD). Alerts describe symptoms and never state the root cause."""

from __future__ import annotations

import random
from dataclasses import dataclass

from tinyops.sim.cluster import ClusterState

ALL = None
# always crash-looping in every variant
CRASHING = frozenset({"crashloop_missing_env", "bad_configmap_value"})
# zero ready replicas in every variant
NO_READY = frozenset({
    "crashloop_missing_env", "image_pull_backoff", "readiness_probe_misconfig",
    "pending_insufficient_memory", "bad_configmap_value", "hung_process_restart",
})
# at least one replica not ready in every variant
SOME_NOT_READY = NO_READY | {"oom_killed", "node_memory_pressure_eviction"}


@dataclass(frozen=True)
class AlertTemplate:
    text: str
    types: frozenset[str] | None = ALL  # None = valid for every failure type


TRAIN_ALERTS: tuple[AlertTemplate, ...] = (
    AlertTemplate("PagerDuty: {app} in namespace {ns} error rate above 5% for 10 minutes."),
    AlertTemplate("Users report {app} is down. Namespace: {ns}. Please investigate."),
    AlertTemplate("Alertmanager: KubePodCrashLooping — pod {pod} ({ns}).", CRASHING),
    AlertTemplate("Deploy of {app} to {ns} finished 5 minutes ago and the service is unhealthy."),
    AlertTemplate("Alertmanager: KubePodNotReady — {app} in {ns} has had no ready pods for 10 "
                  "minutes.", NO_READY),
    AlertTemplate("Latency SLO burn: {app} ({ns}) p99 latency and 5xx rate are elevated. "
                  "Investigate and report the root cause."),
    AlertTemplate("Slack #oncall: customers see errors from {app}. Can someone look at namespace "
                  "{ns}?"),
    AlertTemplate("Synthetic check failed: GET https://{app}.example.com/health returned 503 "
                  "(ns: {ns})."),
    AlertTemplate("Alertmanager: KubeDeploymentReplicasMismatch — deployment {app} in {ns} has "
                  "fewer ready replicas than desired.", SOME_NOT_READY),
)

OOD_ALERTS: tuple[AlertTemplate, ...] = (
    AlertTemplate("[ALERT] {app}/{ns}: availability below target for 15 min (burn rate 14.4)."),
    AlertTemplate("Ticket #4821: {app} is returning errors for some customers (cluster namespace "
                  "{ns})."),
    AlertTemplate("Opsgenie P2: requests to {app} in {ns} are timing out intermittently."),
    AlertTemplate("Status page check red for {app} ({ns}). Please triage."),
    AlertTemplate("On-call handoff note: {app} in {ns} looked flaky overnight, nobody has dug in "
                  "yet."),
)


def make_alert(rng: random.Random, state: ClusterState, ood: bool = False) -> str:
    from tinyops.scenarios.base import faulty_pod, locate_root_cause

    gt = state.ground_truth
    ns, _, _ = locate_root_cause(state)
    pod = faulty_pod(state)
    app = pod.deployment or pod.name
    pool = OOD_ALERTS if ood else TRAIN_ALERTS
    valid = [t for t in pool if t.types is None or gt.failure_type in t.types]
    return rng.choice(valid).text.format(app=app, ns=ns, pod=pod.name)
