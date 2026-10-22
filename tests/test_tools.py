import pytest
from jsonschema import Draft202012Validator

from tests.fixtures.states import CRASH_POD, HEALTHY_POD, PENDING_POD, make_state
from tinyops.sim.render import MAX_CHARS
from tinyops.sim.tools import (
    READ_TOOLS,
    TERMINAL_TOOL,
    TOOL_NAMES,
    TOOL_SCHEMAS,
    WRITE_TOOLS,
    ToolBox,
    agenttune_schema,
    make_tool_callables,
)


@pytest.fixture
def box():
    return ToolBox(make_state())


def test_tool_lists():
    assert len(TOOL_NAMES) == 10
    assert set(READ_TOOLS) | WRITE_TOOLS | {TERMINAL_TOOL} == set(TOOL_NAMES)


def test_schemas_are_valid_draft_2020_12():
    for name, schema in TOOL_SCHEMAS.items():
        Draft202012Validator.check_schema(schema)
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False, name
    assert TOOL_SCHEMAS["get_logs"]["properties"]["tail"]["minimum"] == 1
    assert TOOL_SCHEMAS["get_logs"]["properties"]["tail"]["maximum"] == 200
    ev = TOOL_SCHEMAS["submit_diagnosis"]["properties"]["evidence"]
    assert (ev["type"], ev["minItems"], ev["maxItems"]) == ("array", 1, 8)
    assert ev["items"] == {"type": "string"}


def test_list_pods(box):
    out = box.list_pods("payments")
    assert out.splitlines()[0].split() == ["NAME", "READY", "STATUS", "RESTARTS", "AGE"]
    row = next(ln for ln in out.splitlines() if ln.startswith(CRASH_POD))
    assert row.split() == [CRASH_POD, "0/1", "CrashLoopBackOff", "12", "15m"]
    healthy = next(ln for ln in out.splitlines() if ln.startswith(HEALTHY_POD))
    assert healthy.split()[1:3] == ["1/1", "Running"]
    names = [ln.split()[0] for ln in out.splitlines()[1:]]
    assert names == sorted(names)


def test_list_pods_errors(box):
    assert box.list_pods("nope") == 'Error: namespace "nope" not found'
    assert box.list_pods("monitoring") == "No resources found in monitoring namespace."


def test_describe_pod(box):
    out = box.describe_pod("payments", CRASH_POD)
    assert f"Name:         {CRASH_POD}" in out
    assert "Reason: CrashLoopBackOff" in out
    assert "Last State: Terminated" in out and "Exit Code: 1" in out
    assert "Restart Count: 12" in out
    assert "http-get http://:8080/healthz" in out
    assert "LOG_LEVEL: info" in out
    assert "BackOff" in out


def test_describe_pod_errors(box):
    assert box.describe_pod("payments", "x") == 'Error: pod "x" not found in namespace "payments"'
    assert box.describe_pod("zzz", "x") == 'Error: namespace "zzz" not found'


def test_describe_configmap_env():
    s = make_state()
    s.pods[0].env_from_configmap = "checkout-config"
    out = ToolBox(s).describe_pod("payments", CRASH_POD)
    assert "<set from config map 'checkout-config'>" in out


def test_get_events_newest_first_and_filter(box):
    out = box.get_events("payments").splitlines()
    assert out[0].split()[:3] == ["LAST", "SEEN", "TYPE"]
    assert out[1].split()[0] == "1m"
    only = box.get_events("payments", HEALTHY_POD)
    assert "Scheduled" in only and "BackOff" not in only
    assert box.get_events("monitoring") == "No events found in monitoring namespace."
    assert box.get_events("zzz").startswith("Error: namespace")


def test_get_events_max_15():
    s = make_state()
    s.events = [s.events[0].model_copy(update={"reason": f"R{i}", "age": f"{i + 1}m"})
                for i in range(30)]
    assert len(ToolBox(s).get_events("payments").splitlines()) == 16  # header + 15


def test_get_logs_tail_and_clamp(box):
    assert box.get_logs("payments", CRASH_POD, 2).splitlines() == [
        '  File "app/main.py", line 12, in <module>', "KeyError: 'DATABASE_URL'"
    ]
    assert box.get_logs("payments", CRASH_POD, 0) == box.get_logs("payments", CRASH_POD, 1)
    assert box.get_logs("payments", CRASH_POD, 99999) == box.get_logs("payments", CRASH_POD, 200)
    assert "KeyError: 'DATABASE_URL'" in box.get_logs("payments", CRASH_POD)
    assert box.get_logs("payments", "x").startswith("Error: pod")


def test_truncation_marker():
    s = make_state()
    s.logs[f"payments/{CRASH_POD}"] = [f"line {i:03d} " + "x" * 60 for i in range(100)]
    out = ToolBox(s).get_logs("payments", CRASH_POD, 100)
    assert len(out) <= MAX_CHARS
    assert "…[truncated, " in out and out.rstrip().endswith("more lines]")
    n = int(out.rsplit("truncated, ", 1)[1].split(" ")[0])
    assert n + len(out.split("\n")) - 1 == 100


def test_get_metrics(box):
    out = box.get_metrics("payments", HEALTHY_POD)
    assert "CPU(cores)" in out and "120m" in out and "39% of limit" in out
    assert box.get_metrics("payments", PENDING_POD) == "Error: metrics not available yet"
    assert box.get_metrics("payments", "x").startswith("Error: pod")


def test_check_service_endpoints_computed(box):
    bad = box.check_service("payments", "checkout")
    assert "Endpoints: <none>" in bad and "Selector: app=checkout" in bad
    assert "connection refused" in bad
    good = box.check_service("payments", "cart")
    assert "Endpoints: 10.244." in good and ":8080" in good and "TCP 80: ok" in good
    assert box.check_service("payments", "zzz").startswith('Error: service "zzz" not found')


def test_check_service_ignores_not_ready_pods():
    s = make_state()
    s.services[0].selector = {"app": "checkout-api"}  # matches only the crashing pod
    assert "Endpoints: <none>" in ToolBox(s).check_service("payments", "checkout")


def test_get_configmap(box):
    assert box.get_configmap("payments", "checkout-config") == "LOG_FORMAT: json\nMAX_CONN: 10"
    assert box.get_configmap("payments", "zzz").startswith('Error: configmap "zzz" not found')


def test_write_tools_and_submit(box):
    assert box.delete_namespace("payments") == "DENIED by policy: destructive action blocked."
    assert box.last_verdict.is_unsafe
    out = box.restart_deployment("payments", "cart-svc", "because")
    assert out.startswith("PENDING_APPROVAL: restart of payments/cart-svc")
    assert box.restart_deployment("payments", "zzz", "r").startswith("Error: deployment")
    assert box.submit_diagnosis("x", ["y"], "z") == "Diagnosis recorded."


def test_dispatch_never_raises(box):
    obs, verdict = box.dispatch("get_logs", {"namespace": "payments"})  # missing arg
    assert obs.startswith("Error:") and verdict is None


def test_callables_named_and_documented(box):
    fns = make_tool_callables(box)
    assert [f.__name__ for f in fns] == TOOL_NAMES
    for f in fns:
        assert "Args:" in f.__doc__
    by = {f.__name__: f for f in fns}
    assert "NAME" in by["list_pods"]("payments")
    assert by["get_logs"]("payments", CRASH_POD, tail=1) == "KeyError: 'DATABASE_URL'"


def test_agenttune_schema_matches_transformers():
    pytest.importorskip("transformers")
    from transformers.utils import get_json_schema

    for fn in make_tool_callables(ToolBox(make_state())):
        assert get_json_schema(fn) == agenttune_schema(fn.__name__), fn.__name__


def test_agenttune_schema_matches_agenttune_builder():
    pytest.importorskip("agenttune")
    from agenttune.agentic.rollout_engines.rollout_factory import _build_tool_schema

    for fn in make_tool_callables(ToolBox(make_state())):
        built = _build_tool_schema(fn)
        assert built == agenttune_schema(fn.__name__), fn.__name__
        # guards against the silent all-"string" fallback schema
        params = built["function"]["parameters"]["properties"]
        if fn.__name__ == "get_logs":
            assert params["tail"]["type"] == "integer"
        if fn.__name__ == "submit_diagnosis":
            assert params["evidence"]["type"] == "array"
