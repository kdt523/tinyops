import random
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tinyops.policies.base import ScriptedSequencePolicy
from tinyops.policies.scripted import OraclePolicy
from tinyops.scenarios.build_dataset import load_scenarios
from tinyops.sim.env import run_episode
from tinyops.sim.tools import TOOL_NAMES, ToolCall, agenttune_schema
from tinyops.train import rollout_env as re_
from tinyops.train.rollout_env import (
    ALREADY_SUBMITTED,
    NO_INCIDENT,
    ScenarioRegistry,
    bind,
    make_pre_step_hook,
    make_rollout_tools,
)


@pytest.fixture(scope="module")
def scs(built_data):
    out, seen = [], set()
    for s in load_scenarios("val", built_data):
        if s.alert_text not in seen:
            seen.add(s.alert_text)
            out.append(s)
    return out


@pytest.fixture
def tools():
    t = {f.__name__: f for f in make_rollout_tools()}
    yield t
    bind(None)


def conv(sc):
    return [{"role": "system", "content": "s"}, {"role": "user", "content": sc.alert_text}]


def test_tools_have_the_same_schemas_agenttune_generates_for_the_plain_callables():
    pytest.importorskip("transformers")
    from transformers.utils import get_json_schema

    fns = make_rollout_tools()
    assert [f.__name__ for f in fns] == TOOL_NAMES
    for f in fns:
        assert get_json_schema(f) == agenttune_schema(f.__name__), f.__name__


def test_agenttune_builds_the_same_schemas_for_the_rollout_tools():
    pytest.importorskip("agenttune")
    from agenttune.agentic.rollout_engines.rollout_factory import _build_tool_schema

    for f in make_rollout_tools():
        built = _build_tool_schema(f)
        assert built == agenttune_schema(f.__name__)
        if f.__name__ == "get_logs":
            assert built["function"]["parameters"]["properties"]["tail"]["type"] == "integer"


def test_unbound_tools_say_so(tools):
    bind(None)
    assert tools["list_pods"](namespace="x") == NO_INCIDENT


def test_hook_binds_the_scenario_and_tools_answer_from_it(scs, tools):
    reg, seen = ScenarioRegistry(scs), []
    hook = make_pre_step_hook(reg, seen)
    a, b = scs[0], scs[1]
    hook(0, conv(a), None)
    out_a = tools["list_pods"](namespace=next(iter(a.state.namespaces)))
    hook(0, conv(b), None)
    out_b = tools["list_pods"](namespace=next(iter(b.state.namespaces)))
    assert seen == [a.id, b.id]
    assert out_a != out_b and out_a.startswith(("NAME", "No resources", "Error"))
    hook(0, conv(a), None)  # re-binding a scenario starts a fresh environment
    assert tools["list_pods"](namespace=next(iter(a.state.namespaces))) == out_a


def test_later_iterations_do_not_rebind(scs, tools):
    hook = make_pre_step_hook(ScenarioRegistry(scs))
    hook(0, conv(scs[0]), None)
    first = re_.current()
    hook(3, conv(scs[1]), None)
    assert re_.current() is first


def test_unknown_alert_leaves_tools_unbound(scs, tools):
    hook = make_pre_step_hook(ScenarioRegistry(scs), history := [])
    hook(0, [{"role": "user", "content": "some other alert"}], None)
    assert history == ["<unbound>"] and tools["list_pods"](namespace="x") == NO_INCIDENT


def test_observations_match_the_evaluation_environment_exactly(scs, tools):
    """Same calls -> same text, including errors for bad arguments and unknown namespaces."""
    sc = next(s for s in scs if s.failure_type == "crashloop_missing_env")
    plan = OraclePolicy(sc.state, random.Random(1)).plan
    ns = plan[0].arguments["namespace"]
    extra = [ToolCall(name="get_logs", arguments={"namespace": ns}),                  # missing pod
             ToolCall(name="list_pods", arguments={"namespace": ns, "pod_name": "x"}),  # bad arg name
             ToolCall(name="list_pods", arguments={"namespace": "nope"}),
             ToolCall(name="get_logs", arguments={"namespace": ns, "pod": "p", "tail": 500})]
    calls = [*plan[:-1], *extra, plan[-1]]
    ep = run_episode(sc.state, ScriptedSequencePolicy(calls), sc.alert_text, max_steps=20)
    make_pre_step_hook(ScenarioRegistry(scs))(0, conv(sc), None)
    for call, step in zip(calls, ep.steps, strict=True):
        assert tools[call.name](**call.arguments) == step.observation, call


def test_duplicate_and_post_submit_behaviour(scs, tools):
    sc = scs[0]
    make_pre_step_hook(ScenarioRegistry(scs))(0, conv(sc), None)
    plan = OraclePolicy(sc.state, random.Random(1)).plan
    outs = [tools[c.name](**c.arguments) for c in plan]
    assert outs[-1] == "Diagnosis recorded."
    assert tools["list_pods"](namespace="x") == ALREADY_SUBMITTED


def test_delete_namespace_is_denied_through_the_tools(scs, tools):
    sc = scs[0]
    make_pre_step_hook(ScenarioRegistry(scs))(0, conv(sc), None)
    assert tools["delete_namespace"](namespace="anything").startswith("DENIED by policy")
    assert re_.current().env.steps[-1].is_unsafe


def test_positional_arguments_are_mapped_by_signature(scs, tools):
    sc = scs[0]
    make_pre_step_hook(ScenarioRegistry(scs))(0, conv(sc), None)
    ns = next(iter(sc.state.namespaces))
    assert tools["list_pods"](ns) == tools["list_pods"](namespace=ns)
    assert tools["get_events"](ns, "") .startswith(("LAST", "No events", "Error"))


def test_interleaved_and_concurrent_rollouts_never_share_state(scs, tools):
    """Each thread runs its own rollout (hook + tools in the same context)."""
    hook = make_pre_step_hook(ScenarioRegistry(scs))
    picks = scs[:8]

    def rollout(sc):
        hook(0, conv(sc), None)
        plan = OraclePolicy(sc.state, random.Random(1)).plan
        outs = []
        for c in plan:
            time.sleep(0.001)  # give other threads a chance to interleave
            outs.append(tools[c.name](**c.arguments))
        ep = run_episode(sc.state, ScriptedSequencePolicy(plan), sc.alert_text)
        return [s.observation for s in ep.steps] == outs

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(rollout, picks))


def test_registry_rejects_duplicate_alerts(scs):
    import copy

    twin = copy.deepcopy(scs[0])
    twin.id = "other"
    with pytest.raises(ValueError, match="duplicate alert"):
        ScenarioRegistry([scs[0], twin])
