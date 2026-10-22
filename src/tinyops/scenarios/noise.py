"""Benign log lines, misleading warnings and unrelated events (the noise around a real fault)."""

from __future__ import annotations

import random

from tinyops.sim.cluster import Event

_RESOURCES = ("orders", "items", "users", "carts", "sessions", "quotes", "invoices", "profiles")
_TABLES = ("orders", "sessions", "audit_log", "inventory", "events")
_JOBS = ("metrics-flush", "token-refresh", "cleanup", "reindex", "heartbeat-sync")
_FLAGS = ("new_checkout", "fast_path", "beta_ui", "dark_mode", "async_export")
_KEYS = ("user:{n}", "sku:{n}", "cart:{n}", "quote:{n}")

# Substrings that must never appear in benign noise (they are real key evidence elsewhere).
FORBIDDEN_IN_NOISE = (
    "KeyError", "environment variable", "invalid value", "deadlock", "OOMKilled",
    "connection refused", "listening on :", "Exit Code", "not found", "Insufficient",
    "low on resource", "Failed to pull", "Readiness probe", "DENIED",
)


def _templates(rng: random.Random, app: str, downstream: str) -> list[str]:
    n = rng.randint(1, 9999)
    d = rng.randint(2, 180)
    res = rng.choice(_RESOURCES)
    return [
        f"INFO request completed path=/api/v1/{res} status=200 dur={d}ms",
        f"INFO request completed path=/api/v1/{res}/{n} status=200 dur={d}ms",
        f"INFO request completed path=/api/v1/{res} status=201 dur={d}ms",
        "INFO health check ok",
        "INFO GET /healthz 200 dur=1ms",
        f"INFO cache hit key={rng.choice(_KEYS).format(n=n)} ttl={rng.randint(5, 300)}s",
        f"INFO cache miss key={rng.choice(_KEYS).format(n=n)}",
        f"INFO gc pause {rng.randint(2, 40)}ms heap={rng.randint(40, 180)}MB",
        f"WARN slow query took {rng.randint(8, 20) / 10:.1f}s table={rng.choice(_TABLES)}",
        f"INFO scheduled job {rng.choice(_JOBS)} completed in {d}ms",
        f"INFO connection pool stats active={rng.randint(1, 8)} idle={rng.randint(1, 8)}",
        f"INFO metrics flushed points={rng.randint(20, 400)}",
        "INFO config reloaded (no changes)",
        f"INFO user session refreshed id={n:04x}",
        f"INFO worker heartbeat ok id={rng.randint(1, 8)}",
        f"INFO outbound call to {downstream} ok dur={d}ms",
        f"WARN retrying request to {downstream} attempt=2/3 (succeeded)",
        f"INFO tls certificate valid days_left={rng.randint(20, 300)}",
        f"INFO feature flag evaluated flag={rng.choice(_FLAGS)} value=false",
        f"INFO queue depth={rng.randint(0, 30)}",
        f"DEBUG {app} request id={n:05d} routed to handler",
        f"INFO batch processed items={rng.randint(5, 60)} dur={d}ms",
        f"INFO background sync finished records={rng.randint(10, 500)}",
        f"WARN deprecated API field 'legacy_id' used by client version 1.{rng.randint(0, 9)}",
        f"INFO rate limiter bucket refilled tokens={rng.randint(50, 200)}",
        f"INFO audit event written id={n:06d}",
        f"DEBUG gc cycle {rng.randint(100, 999)} finished",
    ]


def benign_line(rng: random.Random, app: str, downstream: str = "user-directory") -> str:
    return rng.choice(_templates(rng, app, downstream))


def healthy_logs(rng: random.Random, app: str, n: int, downstream: str = "user-directory") -> list[str]:
    return [benign_line(rng, app, downstream) for _ in range(n)]


def misleading_warning(rng: random.Random, downstream: str) -> str:
    """A WARN that looks suspicious but is NOT the root cause (used at `hard`)."""
    return rng.choice(
        [
            f"WARN upstream {downstream} latency high p99={rng.randint(900, 3000)}ms",
            f"WARN high latency calling {downstream} (p99={rng.randint(900, 3000)}ms)",
            f"WARN {downstream} responded slowly dur={rng.randint(1200, 4000)}ms",
        ]
    )


def noise_events(rng: random.Random, ns: str, objects: list[tuple[str, str]], n: int) -> list[Event]:
    """Normal events (Scheduled/Pulled/Created/Started) for other objects, e.g. [("Pod", name)]."""
    out: list[Event] = []
    kinds = [
        ("Scheduled", "Successfully assigned {ns}/{obj} to {node}"),
        ("Pulled", "Container image already present on machine"),
        ("Created", "Created container app"),
        ("Started", "Started container app"),
    ]
    for _ in range(n):
        kind, name = rng.choice(objects)
        reason, msg = rng.choice(kinds)
        out.append(
            Event(
                type="Normal", reason=reason, object_kind=kind, object_name=name, namespace=ns,
                message=msg.format(ns=ns, obj=name, node=f"node-{rng.randint(1, 3)}"),
                age=f"{rng.randint(2, 40)}h",
            )
        )
    return out


def unrelated_warning_event(rng: random.Random, ns: str, obj_name: str) -> Event:
    reason, kind, msg = rng.choice(
        [
            ("FailedGetResourceMetric", "HorizontalPodAutoscaler",
             "failed to get cpu utilization: missing request for cpu"),
            ("FailedMount", "Pod", "MountVolume.SetUp failed for volume \"tmp-cache\": transient timeout (retry succeeded)"),
            ("BackoffLimitExceeded", "Job", "Job has reached the specified backoff limit"),
        ]
    )
    return Event(type="Warning", reason=reason, object_kind=kind, object_name=obj_name,
                 namespace=ns, message=msg, age=f"{rng.randint(3, 20)}h")
