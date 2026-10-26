import itertools
from types import SimpleNamespace

import pytest

from tinyops.policies.api_policy import APIPolicy
from tinyops.policies.fewshot import build_fewshot_episodes
from tinyops.policies.hf_policy import HFPolicy, fewshot_messages, splice_fewshot
from tinyops.policies.parsing import has_complete_call_json
from tinyops.policies.prompt import build_system_prompt

CALL = '[{"tool_name": "list_pods", "parameters": {"namespace": "a"}}]'


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", False),
        ("Let me look", False),
        ('[{"tool_name": "list_pods", "parameters": {"namespace": "a"', False),
        (CALL, True),
        (f"Sure. {CALL}", True),
        ('{"tool_name": "list_pods", "parameters": {}}', True),
        ("see [pod] and {ns} first", False),  # balanced brackets that are not a call
        (f"see [pod] first: {CALL}", True),
        ('[{"tool_name": "x", "parameters": {"a": "]}"}}]', True),  # brackets inside a string
    ],
)
def test_has_complete_call_json(text, expected):
    assert has_complete_call_json(text) is expected


def test_fewshot_episodes_come_from_train_with_distinct_types(built_data):
    eps = build_fewshot_episodes(2, 1234, built_data)
    assert len(eps) == 2 and all(e.scenario_id.startswith("train-") for e in eps)
    assert len({e.scenario_id.split("-")[1] for e in eps}) == 2
    assert all(e.termination == "submitted" for e in eps)
    assert [e.scenario_id for e in eps] == [e.scenario_id for e in build_fewshot_episodes(2, 1234, built_data)]
    assert build_fewshot_episodes(0, 1234, built_data) == []
    assert [e.scenario_id for e in eps] != [e.scenario_id for e in build_fewshot_episodes(2, 99, built_data)]


def test_fewshot_never_uses_injected_or_eval_scenarios(built_data):
    from tinyops.scenarios.build_dataset import load_scenarios

    train = {s.id: s for s in load_scenarios("train", built_data)}
    for seed in range(1, 6):
        for e in build_fewshot_episodes(3, seed, built_data):
            assert not train[e.scenario_id].injected


def test_fewshot_messages_are_spliced_after_the_system_prompt(built_data):
    shots = fewshot_messages(build_fewshot_episodes(2, 1234, built_data))
    assert shots[0]["role"] == "user" and "system" not in {m["role"] for m in shots}
    real = [{"role": "system", "content": "S"}, {"role": "user", "content": "alert"}]
    spliced = splice_fewshot(real, shots)
    assert spliced[0] == real[0] and spliced[-1] == real[1] and len(spliced) == len(shots) + 2
    roles = [m["role"] for m in spliced[1:]]
    assert all(a != b for a, b in itertools.pairwise(roles))  # strictly alternating
    assert splice_fewshot(real, []) is real


def test_hf_policy_constructs_without_loading_a_model(built_data):
    p = HFPolicy("CohereLabs/tiny-aya-global")
    assert p.name == "tiny-aya-global-zeroshot" and p._model is None
    p2 = HFPolicy("CohereLabs/tiny-aya-global", adapter="me/tinyops-tiny-aya-sft", fewshot=build_fewshot_episodes(2, 1, built_data))
    assert p2.name == "tiny-aya-global-tinyops-tiny-aya-sft-fewshot2"
    cfg = p2.config()
    assert cfg["adapter"] == "me/tinyops-tiny-aya-sft" and cfg["decoding"] == "greedy"
    assert cfg["fewshot"] == 2 and cfg["load_in_4bit"] is False


def test_hf_policy_halves_the_batch_on_cuda_oom(monkeypatch):
    torch = pytest.importorskip("torch")
    p = HFPolicy("CohereLabs/tiny-aya-global")
    sizes = []

    def fake_generate(batch):
        sizes.append(len(batch))
        if len(batch) > 2:
            raise torch.cuda.OutOfMemoryError("out of memory")
        return [f"out{m[0]['content']}" for m in batch]

    monkeypatch.setattr(p, "_generate", fake_generate)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    batch = [[{"content": str(i)}] for i in range(5)]
    assert p.act_batch(batch) == [f"out{i}" for i in range(5)]  # order preserved
    assert sizes == [5, 2, 3, 1, 2]
    monkeypatch.setattr(p, "_generate", lambda b: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("x")))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        p.act_batch(batch[:1])  # a single prompt that cannot fit is a real failure


def fake_response(text, p=11, c=7):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage=SimpleNamespace(prompt_tokens=p, completion_tokens=c))


def test_api_policy_parses_and_records_usage():
    pol = APIPolicy("gemini/x", completion_fn=lambda **kw: fake_response(CALL), sleep=lambda s: None)
    out = pol.act([{"role": "user", "content": "hi"}])
    assert out.tool_call.name == "list_pods" and out.text == CALL
    assert out.usage["prompt_tokens"] == 11 and out.usage["completion_tokens"] == 7
    assert out.usage["latency_s"] >= 0 and pol.config()["model_id"] == "gemini/x"


def test_api_policy_passes_greedy_settings_and_messages():
    seen = {}

    def fn(**kw):
        seen.update(kw)
        return fake_response("prose only")

    msgs = [{"role": "system", "content": build_system_prompt()}, {"role": "user", "content": "a"}]
    out = APIPolicy("m", completion_fn=fn, sleep=lambda s: None).act(msgs)
    assert seen["temperature"] == 0 and seen["max_tokens"] == 256 and seen["messages"] == msgs
    assert out.tool_call is None and out.text == "prose only"


def test_api_policy_retries_with_exponential_backoff():
    calls, sleeps = [], []

    def flaky(**kw):
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("503")
        return fake_response(CALL)

    pol = APIPolicy("m", completion_fn=flaky, sleep=sleeps.append, rpm=6000)
    assert pol.act([]).tool_call is not None and len(calls) == 3
    backoffs = [s for s in sleeps if s >= 1.0]
    assert backoffs == [1.0, 2.0]


def test_api_policy_gives_up_after_three_tries():
    def dead(**kw):
        raise TimeoutError("nope")

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        APIPolicy("m", completion_fn=dead, sleep=lambda s: None).act([])


def test_api_policy_rate_limit():
    sleeps = []
    pol = APIPolicy("m", rpm=60, completion_fn=lambda **kw: fake_response(CALL), sleep=sleeps.append)
    pol.act([])
    pol.act([])
    assert any(0.5 < s <= 1.0 for s in sleeps)  # second call waits ~1s at 60 rpm


@pytest.mark.slow
def test_hf_policy_end_to_end_on_a_tiny_model(built_data):
    """Plumbing only (padding, stopping, usage). Needs network for a 135M model; run with -m slow."""
    p = HFPolicy("HuggingFaceTB/SmolLM2-135M-Instruct", max_new_tokens=24)
    msgs = [{"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": "Alert: payments is down."}]
    outs = p.act_batch([msgs, msgs[:1] + [{"role": "user", "content": "Hello"}]])
    assert len(outs) == 2
    for o in outs:
        assert isinstance(o.text, str) and o.usage["completion_tokens"] <= 24
        assert o.usage["prompt_tokens"] > 100 and o.usage["batch_size"] == 2
    import torch

    expected = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if torch.cuda.is_available() else "float32"
    assert p.config()["dtype"] == expected
