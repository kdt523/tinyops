import json

import pytest

from tests.fixtures.states import make_state
from tests.test_env import four_step
from tinyops.policies.base import ScriptedSequencePolicy
from tinyops.sim.env import run_episode
from tinyops.sim.export import (
    conversation_from_episode,
    episode_to_trajectory_line,
    load_with_auditkit,
    write_trajectories,
)
from tinyops.utils.io import read_jsonl


@pytest.fixture
def episode():
    return run_episode(make_state(), ScriptedSequencePolicy(four_step()), "ALERT", scenario_id="v-1")


def test_line_has_required_keys(episode):
    line = episode_to_trajectory_line(episode)
    assert {"task", "steps", "final_response", "reward", "metadata", "trajectory_id"} <= set(line)
    assert line["reward"] is None and line["task"] == "ALERT"
    assert len(line["steps"]) == 4
    call = line["steps"][1]["action"]["tool_calls"][0]
    assert call["name"] == "describe_pod" and set(call) == {"id", "name", "arguments"}
    assert line["steps"][0]["observation"].startswith("NAME")
    md = line["metadata"]
    assert md["scenario_id"] == "v-1" and md["termination"] == "submitted"
    assert "DATABASE_URL" in line["final_response"]
    json.dumps(line)  # serialisable


def test_conversation_matches_what_policy_saw(episode):
    conv = conversation_from_episode(episode)
    assert [m["role"] for m in conv] == ["system", "user"] + ["assistant", "user"] * 3 + ["assistant"]
    assert json.loads(conv[-1]["content"])[0]["tool_name"] == "submit_diagnosis"
    assert conv[3]["content"].startswith("Tool result for list_pods: ")


def test_write_trajectories_roundtrip(tmp_path, episode):
    p = tmp_path / "t.jsonl"
    write_trajectories([episode, episode], p)
    rows = read_jsonl(p)
    assert len(rows) == 2 and rows[0] == episode_to_trajectory_line(episode)


def test_load_with_auditkit_preserves_tool_names(tmp_path, episode):
    pytest.importorskip("auditkit")
    p = tmp_path / "t.jsonl"
    write_trajectories([episode], p)
    eps = load_with_auditkit(p)
    assert len(eps) == 1
    names = [e.payload["name"] for e in eps[0].events if e.type == "tool_call"]
    assert names == ["list_pods", "describe_pod", "get_logs", "submit_diagnosis"]


def test_auditkit_pairs_calls_with_conversation(tmp_path, episode):
    pytest.importorskip("auditkit")
    p = tmp_path / "t.jsonl"
    write_trajectories([episode], p)
    ak_ep = load_with_auditkit(p)[0]
    assert ak_ep.coverage["tool_call_names"] and ak_ep.coverage["tool_call_arguments"]


def test_eventlog_loads_in_auditkit(episode):
    pytest.importorskip("agenttune")
    pytest.importorskip("auditkit")
    from auditkit.agent_eval.importers import episode_from_eventlog

    from tinyops.sim.export import episode_to_eventlog

    log = episode_to_eventlog(episode)
    assert log.tier == "light"
    ak_ep = episode_from_eventlog(log)
    names = [e.payload["name"] for e in ak_ep.events if e.type == "tool_call"]
    assert names == ["list_pods", "describe_pod", "get_logs", "submit_diagnosis"]
