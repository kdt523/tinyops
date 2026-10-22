"""Pure functions that turn cluster state into kubectl-like text. Deterministic ordering."""

from __future__ import annotations

import hashlib

from tinyops.sim.cluster import Event, Pod, PodMetrics, Service, parse_mem

MAX_CHARS = 1500


def truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    """Cut `text` to at most `max_chars`, at a line boundary, with a marker for what was cut."""
    if len(text) <= max_chars:
        return text
    lines = text.split("\n")
    kept: list[str] = []
    used = 0
    for i, line in enumerate(lines):
        marker = f"\n…[truncated, {len(lines) - i} more lines]"
        extra = len(line) + (1 if kept else 0)
        if used + extra + len(marker) > max_chars:
            return "\n".join(kept) + f"\n…[truncated, {len(lines) - len(kept)} more lines]"
        kept.append(line)
        used += extra
    return "\n".join(kept)


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(header)]
    fmt = lambda cols: "  ".join(c.ljust(w) for c, w in zip(cols, widths, strict=True)).rstrip()
    return "\n".join([fmt(header), *(fmt(r) for r in rows)])


def render_pod_table(pods: list[Pod]) -> str:
    rows = [
        [p.name, f"{1 if p.ready else 0}/1", p.phase, str(p.restarts), p.age]
        for p in sorted(pods, key=lambda p: p.name)
    ]
    return _table(["NAME", "READY", "STATUS", "RESTARTS", "AGE"], rows)


def render_events(events: list[Event]) -> str:
    rows = [
        [e.age, e.type, e.reason, f"{e.object_kind.lower()}/{e.object_name}", e.message]
        for e in events
    ]
    return _table(["LAST SEEN", "TYPE", "REASON", "OBJECT", "MESSAGE"], rows)


def _state_lines(label: str, st) -> list[str]:
    out = [f"{label}: {st.state.capitalize()}"]
    if st.reason:
        out.append(f"  Reason: {st.reason}")
    if st.exit_code is not None:
        out.append(f"  Exit Code: {st.exit_code}")
    if st.message:
        out.append(f"  Message: {st.message}")
    return out


def render_describe(pod: Pod, events: list[Event]) -> str:
    lines = [
        f"Name:         {pod.name}",
        f"Namespace:    {pod.namespace}",
        f"Node:         {pod.node or '<none>'}",
        "Labels:       " + (", ".join(f"{k}={v}" for k, v in sorted(pod.labels.items())) or "<none>"),
        f"Image:        {pod.image}",
        *_state_lines("State", pod.current_state),
    ]
    if pod.last_state is not None:
        lines += _state_lines("Last State", pod.last_state)
    r = pod.resources
    lines += [
        f"Ready:        {'True' if pod.ready else 'False'}",
        f"Restart Count: {pod.restarts}",
        f"Limits:       cpu={r.cpu_limit}, memory={r.mem_limit}",
        f"Requests:     cpu={r.cpu_request}, memory={r.mem_request}",
    ]
    for pr in sorted(pod.probes, key=lambda p: p.kind):
        lines.append(f"{pr.kind.capitalize()}:    http-get http://:{pr.port}{pr.path}")
    lines.append("Environment:")
    env_lines = [f"  {k}: {v}" for k, v in sorted(pod.env.items())]
    if pod.env_from_configmap:
        env_lines.append(f"  <set from config map '{pod.env_from_configmap}'>")
    lines += env_lines or ["  <none>"]
    lines.append("Events:")
    recent = events[:5]
    if recent:
        lines += ["  " + ln for ln in render_events(recent).split("\n")]
    else:
        lines.append("  <none>")
    return "\n".join(lines)


def render_logs(lines: list[str], tail: int) -> str:
    return "\n".join(lines[-tail:])


def render_metrics(pod: Pod, m: PodMetrics) -> str:
    limit = parse_mem(pod.resources.mem_limit)
    pct = 100.0 * m.mem_bytes / limit if limit else 0.0
    return "\n".join(
        [
            _table(["NAME", "CPU(cores)", "MEMORY(bytes)"], [[pod.name, f"{m.cpu_millicores}m", str(m.mem_bytes)]]),
            f"memory usage: {pct:.0f}% of limit ({pod.resources.mem_limit})",
        ]
    )


def pod_ip(pod: Pod) -> str:
    h = hashlib.sha256(f"{pod.namespace}/{pod.name}".encode()).digest()
    return f"10.244.{h[0] % 250 + 1}.{h[1] % 250 + 1}"


def render_service(svc: Service, endpoints: list[Pod], dns_ok: bool = True) -> str:
    ep = ", ".join(f"{pod_ip(p)}:{svc.target_port}" for p in sorted(endpoints, key=lambda p: p.name))
    tcp = "ok" if endpoints else "connection refused"
    return "\n".join(
        [
            f"Name: {svc.name}",
            f"Namespace: {svc.namespace}",
            "Selector: " + (", ".join(f"{k}={v}" for k, v in sorted(svc.selector.items())) or "<none>"),
            f"Port: {svc.port} -> {svc.target_port}",
            f"Endpoints: {ep or '<none>'}",
            f"DNS: {'ok' if dns_ok else 'failed'} | TCP {svc.port}: {tcp}",
        ]
    )
