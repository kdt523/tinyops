"""Score an episodes.jsonl file against its split.

    python -m tinyops.eval.score_episodes --episodes results/episodes/<run>.jsonl \
        --split test --out results/metrics/<run>.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tinyops.scenarios.base import Scenario
from tinyops.scoring.metrics import GROUP_KEYS, EpisodeScore, aggregate, score_episode
from tinyops.sim.env import Episode
from tinyops.utils.io import read_jsonl, write_json, write_jsonl
from tinyops.utils.log import get_logger
from tinyops.utils.paths import DATA

log = get_logger(__name__)

TABLE_METRICS = [
    ("diagnosis_success", "Diagnosis success", True),
    ("grounding", "Grounding", True),
    ("format_valid_rate", "Format valid", True),
    ("schema_valid_rate", "Schema valid", True),
    ("tool_selection_precision", "Tool-selection precision", True),
    ("hallucinated_tools_per_ep", "Hallucinated tools / ep", False),
    ("redundant_calls_per_ep", "Redundant calls / ep", False),
    ("loop_rate", "Loop rate", True),
    ("steps_to_submit", "Steps to submit", False),
    ("unnecessary_write_rate", "Unnecessary-write rate", True),
    ("unsafe_rate", "Unsafe rate (attacks)", True),
    ("injection_robust_success", "Injection-robust success", True),
    ("over_refusal_rate", "Over-refusal rate", True),
    ("correct_fix_rate", "Correct-fix rate (hung)", True),
    ("tokens_per_ep", "Tokens / ep", False),
    ("latency_per_ep_s", "Latency / ep (s)", False),
]


def fmt(stat: dict, pct: bool) -> str:
    if stat["n"] == 0 or stat["mean"] is None:
        return "n/a"
    scale = 100 if pct else 1
    unit = "%" if pct else ""
    prec = 1
    return (f"{stat['mean'] * scale:.{prec}f}{unit} "
            f"[{stat['lo'] * scale:.{prec}f}, {stat['hi'] * scale:.{prec}f}] (n={stat['n']})")


def markdown_table(title: str, agg: dict) -> str:
    rows = ["| Metric | Value [95% CI] |", "|---|---|"]
    rows += [f"| {label} | {fmt(agg['headline'][key], pct)} |" for key, label, pct in TABLE_METRICS]
    return f"### {title}\n\n" + "\n".join(rows) + "\n"


def load_split_scenarios(split: str, data: Path = DATA) -> dict[str, Scenario]:
    return {r["id"]: Scenario.model_validate(r) for r in read_jsonl(data / split / "scenarios.jsonl")}


def score_file(episodes_path: Path, split: str, out: Path, data: Path = DATA) -> dict:
    scenarios = load_split_scenarios(split, data)
    episodes = [Episode.model_validate(r) for r in read_jsonl(episodes_path)]
    unknown = [e.scenario_id for e in episodes if e.scenario_id not in scenarios]
    if unknown:
        raise ValueError(f"{len(unknown)} episodes are not in split {split!r} (e.g. {unknown[:3]})")
    scores: list[EpisodeScore] = [score_episode(e, scenarios[e.scenario_id]) for e in episodes]
    run = out.stem
    result = {
        "run": run, "split": split, "episodes_file": str(episodes_path),
        "n_scenarios_in_split": len(scenarios), "n_episodes": len(scores),
        "complete": len(scores) == len(scenarios),
        "policies": sorted({s.policy for s in scores}),
        **aggregate(scores, by=list(GROUP_KEYS)),
    }
    write_json(out, result)
    write_jsonl(out.with_suffix(".scores.jsonl"), (s.model_dump() for s in scores))
    out.with_suffix(".md").write_text(
        markdown_table(f"{run} on {split} (n={len(scores)})", result), encoding="utf-8")
    if not result["complete"]:
        log.warning("%s covers %d of %d scenarios in %s", run, len(scores), len(scenarios), split)
    return result


def main(
    episodes: Annotated[Path, typer.Option(help="episodes .jsonl (Episode per line)")],
    split: Annotated[str, typer.Option(help="split the scenarios come from")],
    out: Annotated[Path, typer.Option(help="metrics .json to write (+ .md, .scores.jsonl)")],
    data: Annotated[Path, typer.Option(help="data directory")] = DATA,
) -> None:
    score_file(episodes, split, out, data)
    log.info("wrote %s", out)


if __name__ == "__main__":
    typer.run(main)
