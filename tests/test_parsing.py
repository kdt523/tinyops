import pytest

from tinyops.policies.base import format_tool_call_text
from tinyops.policies.parsing import balanced_end, parse_tool_call
from tinyops.sim.tools import ToolCall

LIST = '[{"tool_name": "describe_pod", "parameters": {"namespace": "a", "pod": "p"}}]'


def ok(text, fmt, name="describe_pod", args=None, n=1):
    r = parse_tool_call(text)
    assert r.ok, (r.error, text)
    assert r.format == fmt and r.call.name == name and r.n_calls_found == n
    if args is not None:
        assert r.call.arguments == args
    return r


def fails(text, kind):
    r = parse_tool_call(text)
    assert not r.ok and r.call is None and r.error_kind == kind and r.error, (r, text)
    return r


def test_fallback_list():
    ok(LIST, "fallback_list", args={"namespace": "a", "pod": "p"})


def test_roundtrip_with_formatter():
    call = ToolCall(name="get_logs", arguments={"namespace": "a", "pod": "p", "tail": 20})
    r = parse_tool_call(format_tool_call_text(call))
    assert r.ok and r.call == call and r.format == "fallback_list"


def test_cohere_action_block():
    ok(f"<|START_ACTION|>{LIST}<|END_ACTION|>", "cohere_action", args={"namespace": "a", "pod": "p"})


def test_cohere_action_without_end_marker():
    ok(f"<|START_ACTION|>\n{LIST}", "cohere_action")


def test_cohere_action_with_reasoning_around_it():
    ok(f"<|START_THINKING|>hmm [1]<|END_THINKING|><|START_ACTION|>{LIST}<|END_ACTION|>",
       "cohere_action")


def test_single_json_object():
    ok('{"tool_name": "list_pods", "parameters": {"namespace": "a"}}', "json_object", "list_pods")


def test_fenced_variants():
    ok(f"```json\n{LIST}\n```", "fallback_list_fenced")
    ok('```\n{"tool_name": "list_pods", "parameters": {"namespace": "a"}}\n```',
       "json_object_fenced", "list_pods")
    ok(f"Sure!\n```JSON\n{LIST}\n```\nDone.", "fallback_list_fenced")


@pytest.mark.parametrize("name_key", ["tool_name", "name", "tool"])
@pytest.mark.parametrize("param_key", ["parameters", "arguments", "args"])
def test_key_aliases(name_key, param_key):
    text = f'{{"{name_key}": "list_pods", "{param_key}": {{"namespace": "a"}}}}'
    ok(text, "json_object", "list_pods", {"namespace": "a"})


def test_prose_before_and_after():
    ok(f"I will inspect the pod now. {LIST} Let me know what you see.", "fallback_list")


def test_prose_with_unrelated_brackets_before_call():
    ok(f"Checking [pod] in {{namespace}} and list [1, 2]: {LIST}", "fallback_list")


def test_two_calls_in_one_list_takes_first():
    text = ('[{"tool_name": "list_pods", "parameters": {"namespace": "a"}}, '
            '{"tool_name": "get_events", "parameters": {"namespace": "a"}}]')
    r = ok(text, "fallback_list", "list_pods", n=2)
    assert r.call.arguments == {"namespace": "a"}


def test_two_separate_values_counted():
    text = (f'{LIST}\nthen\n{{"tool_name": "list_pods", "parameters": {{"namespace": "a"}}}}')
    ok(text, "fallback_list", n=2)


def test_nested_braces_and_brackets_inside_strings():
    text = ('[{"tool_name": "submit_diagnosis", "parameters": {"root_cause": "x } ] { [ y", '
            '"evidence": ["a \\"quoted\\" ] line", "b"], "recommended_fix": "z"}}]')
    r = ok(text, "fallback_list", "submit_diagnosis")
    assert r.call.arguments["evidence"][0] == 'a "quoted" ] line'


def test_missing_parameters_become_empty_dict():
    ok('[{"tool_name": "list_pods"}]', "fallback_list", "list_pods", {})


def test_malformed_json_is_not_repaired():
    fails('[{"tool_name": "list_pods", "parameters": {"namespace": "a"},}]', "malformed_json")
    fails("[{'tool_name': 'list_pods', 'parameters': {'namespace': 'a'}}]", "no_tool_call")
    fails('[{"tool_name": "list_pods", "parameters": {"namespace": "a"}}', "malformed_json")
    fails('[{"tool_name": "list_pods" "parameters": {}}]', "malformed_json")


def test_missing_comma_between_two_calls_does_not_leak_the_first():
    text = ('[{"tool_name": "list_pods", "parameters": {"namespace": "a"}} '
            '{"tool_name": "get_events", "parameters": {"namespace": "a"}}]')
    fails(text, "malformed_json")


def test_truncated_output_is_a_failure():
    fails('[{"tool_name": "get_logs", "parameters": {"namespace": "a", "pod": "p", "tail":', "malformed_json")


def test_empty_and_whitespace():
    fails("", "empty_output")
    fails("   \n ", "empty_output")
    fails(None, "empty_output")


def test_prose_only():
    fails("I think the pod is crashing because of a bad config.", "no_tool_call")
    fails("Use list_pods with namespace=payments", "no_tool_call")


def test_json_that_is_not_a_call():
    fails('{"namespace": "payments"}', "no_tool_call")
    fails("[1, 2, 3]", "no_tool_call")
    fails('["list_pods"]', "no_tool_call")


def test_non_dict_parameters_fail():
    fails('[{"tool_name": "list_pods", "parameters": "payments"}]', "invalid_structure")
    fails('[{"tool_name": "list_pods", "parameters": ["payments"]}]', "invalid_structure")
    fails('[{"tool_name": "list_pods", "parameters": "{\\"namespace\\": \\"a\\"}"}]',
          "invalid_structure")


def test_bad_tool_names():
    fails('[{"tool_name": "", "parameters": {}}]', "invalid_structure")
    fails('[{"tool_name": 5, "parameters": {}}]', "invalid_structure")


def test_second_element_must_also_be_a_call():
    fails('[{"tool_name": "list_pods", "parameters": {}}, 7]', "no_tool_call")


def test_hallucinated_tool_still_parses():
    r = ok('[{"tool_name": "kubectl_exec", "parameters": {"cmd": "ls"}}]', "fallback_list",
           "kubectl_exec")
    assert r.call.arguments == {"cmd": "ls"}


def test_balanced_end():
    assert balanced_end("[{}]", 0) == 4
    assert balanced_end('{"a": "}"} tail', 0) == 10
    assert balanced_end("[{]", 0) is None
    assert balanced_end("[1, 2", 0) is None
