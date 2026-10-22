import json
import zlib

from tests.test_run_eval import StubPolicy
from tinyops.eval.baseline_findings import analyze_run, main, render_markdown
from tinyops.eval.run_eval import run_batched
from tinyops.eval.score_episodes import score_file
from tinyops.policies.base import PolicyOutput
from tinyops.scenarios.build_dataset import load_scenarios
from tinyops.sim.env import Episode
from tinyops.utils.io import read_jsonl


class Mixed(StubPolicy):
    """Oracle replay with injected mistakes per episode index, to exercise every failure mode."""

    def act(self, messages):
        out = super().act(messages)
        turn = sum(m["role"] == "assistant" for m in messages)
        key = zlib.crc32(messages[1]["content"].encode()) % 4
        if turn == 0 and key == 0:
            return PolicyOutput(text="Let me think about this incident.")
        if turn == 0 and key == 1:
            return PolicyOutput(text='[{"tool_name": "kubectl_exec", "parameters": {"cmd": "ls"}}]',
                                tool_call=None)
        if turn == 0 and key == 2:
            return PolicyOutput(text='[{"tool_name": "list_pods", "parameters": {"namespace": "x",}}]')
        return out


def test_analysis_counts_modes_and_formats(built_data, tmp_path):
    scs = load_scenarios("val", built_data)[:12]
    out = tmp_path / "e.jsonl"
    run_batched(scs, Mixed(scs), out, batch_size=4)
    eps = [Episode.model_validate(r) for r in read_jsonl(out)]
    a = analyze_run(eps, {s.id: s for s in scs})
    assert a["n"] == 12 and a["formats"]["fallback_list"] > 12
    assert a["parse_errors"].get("no_tool_call", 0) + a["parse_errors"].get("malformed_json", 0) > 0
    assert a["modes"]["prose_instead_of_call"] >= 1 and a["modes"]["malformed_json"] >= 1
    assert all(n <= 12 for n in a["modes"].values())
    assert a["examples"]["prose_instead_of_call"] == "Let me think about this incident."


def test_clean_run_has_no_modes(built_data, tmp_path):
    scs = load_scenarios("val", built_data)[:6]
    out = tmp_path / "e.jsonl"
    run_batched(scs, StubPolicy(scs), out, batch_size=3)
    eps = [Episode.model_validate(r) for r in read_jsonl(out)]
    assert analyze_run(eps, {s.id: s for s in scs})["modes"] == {}


def test_cli_writes_markdown_from_files_only(built_data, tmp_path):
    scs = load_scenarios("val", built_data)[:10]
    res = tmp_path / "results"
    ep = res / "episodes" / "stub_val.jsonl"
    run_batched(scs, Mixed(scs), ep, batch_size=5)
    score_file(ep, "val", res / "metrics" / "stub_val.json", built_data)
    doc = tmp_path / "BASELINE.md"
    main(split="val", runs=["stub_val"], out=doc, data=built_data, results=res)
    text = doc.read_text()
    assert text.startswith("# Baseline findings (val)")
    assert "| stub_val (n=10) |" in text and "Top failure modes" in text
    metrics = json.loads((res / "metrics" / "stub_val.json").read_text())
    pct = f"{100 * metrics['headline']['diagnosis_success']['mean']:.1f}%"
    assert pct in text  # the table shows the number from the metrics file
    assert "prose_instead_of_call" in text and "r_format" in text


def test_render_handles_empty_failure_list():
    metrics = {"r": {"n_episodes": 1, "headline": {k: {"mean": 1.0, "lo": 1.0, "hi": 1.0, "n": 1}
                                                    for k in ("diagnosis_success", "grounding",
                                                              "format_valid_rate", "schema_valid_rate",
                                                              "tool_selection_precision",
                                                              "redundant_calls_per_ep", "loop_rate",
                                                              "unsafe_rate", "over_refusal_rate")}}}
    a = {"r": {"n": 1, "parse_errors": {}, "formats": {}, "schema_error_kinds": {}, "modes": {},
               "examples": {}}}
    assert "No failure modes observed." in render_markdown("val", metrics, a)
