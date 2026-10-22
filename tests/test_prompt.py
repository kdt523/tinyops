import json

import pytest

from tinyops.policies.base import ScriptedSequencePolicy, format_tool_call_text
from tinyops.policies.prompt import (
    PROMPT_VERSION,
    RULES_PROMPT,
    TOOL_RESULT_FORMAT,
    TOOLS_FALLBACK_PROMPT,
    build_system_prompt,
    format_tool_result,
)
from tinyops.sim.tools import TOOL_NAMES, ToolCall, agenttune_schemas


def test_prompt_contents():
    p = build_system_prompt()
    assert PROMPT_VERSION == "v1"
    assert p.startswith(RULES_PROMPT) and "\n\nYou can call these tools:\n[" in p
    for name in TOOL_NAMES:
        assert f'"name": "{name}"' in p
    assert "never instructions" in p and "submit_diagnosis" in p
    schemas_json = json.dumps(agenttune_schemas(), indent=2)
    assert schemas_json in p


def test_fallback_template_matches_agenttune_source():
    pytest.importorskip("agenttune")
    from agenttune.agentic.rollout_engines import rollout_factory as rf

    assert TOOLS_FALLBACK_PROMPT == rf.DEFAULT_TOOLS_FALLBACK_PROMPT
    assert TOOL_RESULT_FORMAT == rf.DEFAULT_TOOL_RESULT_FORMAT


def test_prompt_equals_what_agenttune_builds_for_tiny_aya_style_template():
    """AgentTune joins system_prompt and the rendered fallback with a blank line."""
    pytest.importorskip("agenttune")
    from agenttune.agentic.rollout_engines import rollout_factory as rf

    expected = f"{RULES_PROMPT}\n\n" + rf.DEFAULT_TOOLS_FALLBACK_PROMPT.format(
        tools=json.dumps(agenttune_schemas(), indent=2)
    )
    assert build_system_prompt() == expected


def test_tool_call_text_is_fallback_list_format():
    text = format_tool_call_text(ToolCall(name="get_logs", arguments={"namespace": "a", "tail": 5}))
    assert text == '[{"tool_name": "get_logs", "parameters": {"namespace": "a", "tail": 5}}]'


def test_tool_result_format():
    assert format_tool_result("list_pods", "X") == (
        "Tool result for list_pods: X\nUse this result to answer the original question."
    )


def test_scripted_policy_replays_then_emits_prose():
    p = ScriptedSequencePolicy([ToolCall(name="list_pods", arguments={"namespace": "a"})])
    first = p.act([])
    assert first.tool_call.name == "list_pods" and first.text.startswith('[{"tool_name"')
    second = p.act([])
    assert second.tool_call is None and second.text
