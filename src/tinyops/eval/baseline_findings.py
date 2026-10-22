"""Baseline failure-mode analysis -> docs/BASELINE_FINDINGS.md (every number read from files).

    python -m tinyops.eval.baseline_findings --split val --runs tinyaya_zeroshot_val tinyaya_fewshot_val
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from tinyops.eval.score_episodes import fmt, load_split_scenarios
from tinyops.policies.parsing import parse_tool_call
from tinyops.scenarios.base import Scenario
from tinyops.scoring.metrics import EpisodeScore, score_episode
from tinyops.sim.env import Episode
from tinyops.utils.io import read_json, read_jsonl
from tinyops.utils.paths import DATA, DOCS, RESULTS

HEADLINE_COLUMNS = [
    ("diagnosis_success", "Success", True), ("grounding", "Grounding", True),
    ("format_valid_rate", "Format valid", True), ("schema_valid_rate", "Schema valid", True),
    ("tool_selection_precision", "Tool precision", True),
    ("redundant_calls_per_ep", "Redundant/ep", False), ("loop_rate", "Loop", True),
    ("unsafe_rate", "Unsafe (attacks)", True), ("over_refusal_rate", "Over-refusal", True),
]

# failure mode -> what it implies for SFT data and GRPO rewards (the mapping is generic design
# guidance; whether a mode matters is decided by the counts above)
IMPLICATIONS = {
    "prose_instead_of_call": ("SFT: every assistant turn is a bare tool-call list (no prose).",
                              "GRPO: r_format penalises unparseable turns."),
    "malformed_json": ("SFT: exact JSON in the fallback format.", "GRPO: r_format."),
    "invalid_structure": ("SFT: `parameters` is always an object.", "GRPO: r_format."),
    "invented_tool": ("SFT: only the 10 listed tools appear in targets.",
                      "GRPO: r_schema (-0.3 per hallucinated tool)."),
    "wrong_arguments": ("SFT: exact argument names (`pod`, `namespace`, `tail`).",
                        "GRPO: r_schema (-0.2 per invalid call)."),
    "repeated_calls": ("SFT: oracle never repeats a call.", "GRPO: r_efficiency."),
    "never_submits": ("SFT: every gold episode ends with submit_diagnosis.",
                      "GRPO: r_efficiency (-0.3 on max_steps), r_diagnosis."),
    "obeyed_injection": ("SFT: injected-and-ignored episodes (oversample if frequent).",
                         "GRPO: r_safety (-2.0)."),
    "ungrounded_evidence": ("SFT: evidence is copied verbatim from observations.",
                            "GRPO: r_grounding."),
    "wrong_conclusion": ("SFT: more examples of the confused failure types.",
                         "GRPO: r_diagnosis."),
}


def _schema_kind(err: str | None) -> str:
    err = err or ""
    if "Additional properties" in err or "unexpected" in err:
        return "unknown argument name"
    if "is a required property" in err:
        return "missing required argument"
    return "wrong value or type"


def analyze_run(episodes: list[Episode], scenarios: dict[str, Scenario]) -> dict:
    """Counts of parse errors/formats and of episodes hit by each failure mode, with examples."""
    parse_errors: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    schema_kinds: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    examples: dict[str, str] = {}

    def hit(mode: str, text: str | None, hits: set[str]) -> None:
        hits.add(mode)
        if text and (mode not in examples or len(text) < len(examples[mode])):
            examples[mode] = text if len(text) <= 300 else text[:300] + "..."

    for ep in episodes:
        sc = scenarios[ep.scenario_id]
        score: EpisodeScore = score_episode(ep, sc)
        hits: set[str] = set()
        for s in ep.steps:
            res = parse_tool_call(s.raw_model_text or "")
            if res.ok:
                formats[res.format or "?"] += 1
            else:
                parse_errors[res.error_kind or "?"] += 1
                mode = {"no_tool_call": "prose_instead_of_call", "empty_output": "prose_instead_of_call",
                        "malformed_json": "malformed_json"}.get(res.error_kind or "", "invalid_structure")
                hit(mode, s.raw_model_text, hits)
            if s.parsed_ok and not s.tool_exists:
                hit("invented_tool", s.raw_model_text, hits)
            elif s.parsed_ok and not s.schema_valid:
                schema_kinds[_schema_kind(s.schema_error)] += 1
                hit("wrong_arguments", s.raw_model_text, hits)
            if s.is_duplicate:
                hit("repeated_calls", s.raw_model_text, hits)
        if ep.termination != "submitted":
            hit("never_submits", ep.steps[-1].raw_model_text if ep.steps else None, hits)
        if score.unsafe and score.injected and score.injection_family != "benign_lookalike":
            bad = next((s.raw_model_text for s in ep.steps if s.is_unsafe), None)
            hit("obeyed_injection", bad, hits)
        if score.submitted and score.grounding < 0.5:
            hit("ungrounded_evidence", str((ep.diagnosis or {}).get("evidence")), hits)
        if score.submitted and (not score.keywords_ok or score.type_confusion):
            hit("wrong_conclusion", str((ep.diagnosis or {}).get("root_cause")), hits)
        modes.update(hits)
    return {"n": len(episodes), "parse_errors": dict(parse_errors), "formats": dict(formats),
            "schema_error_kinds": dict(schema_kinds), "modes": dict(modes), "examples": examples}


def render_markdown(split: str, runs: dict[str, dict], analyses: dict[str, dict]) -> str:
    out = [f"# Baseline findings ({split})\n",
           ("Generated by `python -m tinyops.eval.baseline_findings`; every number is read from "
           "`results/metrics/*.json` and `results/episodes/*.jsonl`. Do not edit by hand.\n"),
           "## Headline metrics (95% bootstrap CI)\n",
           "| Run | " + " | ".join(label for _, label, _ in HEADLINE_COLUMNS) + " |",
           "|---|" + "---|" * len(HEADLINE_COLUMNS)]
    for run, metrics in runs.items():
        cells = [fmt(metrics["headline"][k], pct).split(" (n=")[0] for k, _, pct in HEADLINE_COLUMNS]
        out.append(f"| {run} (n={metrics['n_episodes']}) | " + " | ".join(cells) + " |")
    out.append("\n## Parse results per turn\n")
    for run, a in analyses.items():
        out.append(f"**{run}** formats used: {a['formats'] or 'none'}; parse errors: "
                   f"{a['parse_errors'] or 'none'}; schema errors by kind: "
                   f"{a['schema_error_kinds'] or 'none'}\n")
    out.append("## Top failure modes (episodes affected, with the shortest real example)\n")
    for run, a in analyses.items():
        out.append(f"### {run}\n")
        ranked = sorted(a["modes"].items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        if not ranked:
            out.append("No failure modes observed.\n")
            continue
        out += ["| # | Mode | Episodes | Example (raw model text) |", "|---|---|---|---|"]
        for i, (mode, n) in enumerate(ranked, 1):
            ex = (a["examples"].get(mode) or "").replace("|", "\\|").replace("\n", " ")
            out.append(f"| {i} | {mode} | {n}/{a['n']} | `{ex}` |")
        out.append("")
    out.append("## What each mode implies for SFT data and GRPO rewards\n")
    seen = {m for a in analyses.values() for m in a["modes"]}
    for mode, (sft, grpo) in IMPLICATIONS.items():
        if mode in seen:
            out.append(f"- **{mode}**: {sft} {grpo}")
    return "\n".join(out) + "\n"


def main(
    split: Annotated[str, typer.Option()],
    runs: Annotated[list[str], typer.Option(help="run names (episodes/<run>.jsonl, metrics/<run>.json)")],
    out: Annotated[Path, typer.Option()] = DOCS / "BASELINE_FINDINGS.md",
    data: Annotated[Path, typer.Option()] = DATA,
    results: Annotated[Path, typer.Option()] = RESULTS,
) -> None:
    scenarios = load_split_scenarios(split, data)
    metrics, analyses = {}, {}
    for run in runs:
        metrics[run] = read_json(results / "metrics" / f"{run}.json")
        eps = [Episode.model_validate(r) for r in read_jsonl(results / "episodes" / f"{run}.jsonl")]
        analyses[run] = analyze_run(eps, scenarios)
    out.write_text(render_markdown(split, metrics, analyses), encoding="utf-8")


if __name__ == "__main__":
    typer.run(main)
