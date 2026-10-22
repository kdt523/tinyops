"""Hand-built ClusterState fixtures: a healthy namespace, a CrashLoop pod, a mismatched service."""

from __future__ import annotations

from tinyops.sim.cluster import (
    ClusterState,
    ConfigMap,
    ContainerState,
    Deployment,
    Event,
    Fix,
    GroundTruth,
    Node,
    Pod,
    PodMetrics,
    Probe,
    Resources,
    Service,
)

CRASH_POD = "checkout-api-7d9f4c-x2k9z"
HEALTHY_POD = "cart-svc-5b8d6f-q7w2e"
PENDING_POD = "report-job-6c4b8a-m3n4p"


def _pod(name, deployment, app, phase, ready, restarts, cur, last=None, env=None, ns="payments"):
    return Pod(
        name=name, namespace=ns, deployment=deployment, labels={"app": app}, phase=phase,
        ready=ready, restarts=restarts, age="2d" if phase == "Running" else "15m",
        image=f"registry.acme.io/{deployment}:1.4.2", node="node-1" if phase != "Pending" else None,
        resources=Resources(), current_state=cur, last_state=last,
        probes=[Probe(kind="readiness", path="/healthz", port=8080, passing=ready)],
        env=env or {},
    )


def make_state(**gt_overrides) -> ClusterState:
    gt = {
        "failure_type": "crashloop_missing_env",
        "root_cause_object": f"pod/{CRASH_POD}",
        "root_cause_keywords": [["env", "environment variable", "variable"], ["DATABASE_URL"]],
        "key_evidence": ["KeyError: 'DATABASE_URL'"],
        "relevant_tools": ["list_pods", "describe_pod", "get_logs"],
        "correct_fix": None,
    }
    gt.update(gt_overrides)
    crash_logs = [f"INFO request completed path=/api/v1/item/{i} status=200 dur=12ms" for i in range(6)]
    crash_logs += [
        "Traceback (most recent call last):",
        "  File \"app/main.py\", line 12, in <module>",
        "KeyError: 'DATABASE_URL'",
    ]
    return ClusterState(
        cluster_name="kind-prod",
        namespaces=["payments", "monitoring"],
        pods=[
            _pod(CRASH_POD, "checkout-api", "checkout-api", "CrashLoopBackOff", False, 12,
                 ContainerState(state="waiting", reason="CrashLoopBackOff"),
                 ContainerState(state="terminated", reason="Error", exit_code=1),
                 env={"LOG_LEVEL": "info"}),
            _pod(HEALTHY_POD, "cart-svc", "cart-svc", "Running", True, 0,
                 ContainerState(state="running")),
            _pod(PENDING_POD, "report-job", "report-job", "Pending", False, 0,
                 ContainerState(state="waiting", reason="Unschedulable")),
        ],
        deployments=[
            Deployment(name="checkout-api", namespace="payments", replicas=1, ready_replicas=0,
                       image="registry.acme.io/checkout-api:1.4.2",
                       selector={"app": "checkout-api"}, configmap="checkout-config"),
            Deployment(name="cart-svc", namespace="payments", replicas=1, ready_replicas=1,
                       image="registry.acme.io/cart-svc:1.4.2", selector={"app": "cart-svc"}),
        ],
        services=[
            Service(name="checkout", namespace="payments", selector={"app": "checkout"},
                    port=80, target_port=8080),
            Service(name="cart", namespace="payments", selector={"app": "cart-svc"},
                    port=80, target_port=8080),
        ],
        nodes=[Node(name="node-1")],
        events=[
            Event(type="Normal", reason="Pulled", object_kind="Pod", object_name=CRASH_POD,
                  namespace="payments", message="Container image already present", age="10m"),
            Event(type="Warning", reason="BackOff", object_kind="Pod", object_name=CRASH_POD,
                  namespace="payments", message="Back-off restarting failed container",
                  count=30, age="1m"),
            Event(type="Normal", reason="Scheduled", object_kind="Pod", object_name=HEALTHY_POD,
                  namespace="payments", message="Successfully assigned payments/cart", age="2d"),
        ],
        configmaps=[ConfigMap(name="checkout-config", namespace="payments",
                              data={"MAX_CONN": "10", "LOG_FORMAT": "json"})],
        logs={f"payments/{CRASH_POD}": crash_logs,
              f"payments/{HEALTHY_POD}": ["INFO listening on :8080"]},
        metrics={f"payments/{CRASH_POD}": PodMetrics(cpu_millicores=5, mem_bytes=50 * 1024**2),
                 f"payments/{HEALTHY_POD}": PodMetrics(cpu_millicores=120,
                                                       mem_bytes=100 * 1024**2)},
        ground_truth=GroundTruth(**gt),
    )


def make_hung_state() -> ClusterState:
    """State whose correct fix is restarting payments/cart-svc."""
    return make_state(
        failure_type="hung_process_restart",
        correct_fix=Fix(tool="restart_deployment",
                        arguments={"namespace": "payments", "deployment": "cart-svc"}),
    )
