"""Optional teacher data: a closed model solves EXTRA train-split scenarios; keep only clean wins.

    python -m tinyops.train.teacher_data --model gemini/gemini-3.6-flash --n 300 --seed 5678

Extra scenarios are generated with a new seed into data/train_extra/ (ids `train_extra-...`,
split `train`), so they can never overlap val/test. Kept episodes: success, no unsafe action, no
redundant call, grounding 1.0, one plain fallback-list call per turn. Assistant turns are
rewritten to the canonical call text so teacher style does not leak into the targets.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from tinyops.eval.run_eval import run_batched
from tinyops.policies.base import format_tool_call_text
from tinyops.policies.parsing import parse_tool_call
from tinyops.scenarios.base import Scenario
from tinyops.scenarios.build_dataset import build_split, load_scenarios
from tinyops.scoring.metrics import score_episode
from tinyops.sim.env import Episode
from tinyops.sim.tools import ToolCall
from tinyops.train.sft_data import episode_row
from tinyops.utils.io import read_jsonl, write_json, write_jsonl
from tinyops.utils.log import get_logger
from tinyops.utils.paths import DATA

log = get_logger(__name__)


def build_extra_train(seed: int, data: Path = DATA) -> list[Scenario]:
    scenarios = build_split("train", seed, name="train_extra")
    out = data / "train_extra" / "scenarios.jsonl"
    write_jsonl(out, (s.model_dump(mode="json") for s in scenarios))
    return scenarios


def rejection_reason(ep: Episode, sc: Scenario) -> str | None:
    """Why an episode is not clean teacher data (None = keep)."""
    s = score_episode(ep, sc)
    if not s.success:
        return "not_successful"
    if s.unsafe:
        return "unsafe"
    if s.redundant_calls:
        return "redundant_calls"
    if s.grounding < 1.0:
        return "ungrounded"
    for step in ep.steps:
        r = parse_tool_call(step.raw_model_text or "")
        if not r.ok or r.format != "fallback_list" or r.n_calls_found != 1:
            return "not_plain_fallback_format"
    return None


def canonicalize(ep: Episode) -> Episode:
    """Same episode with each assistant turn rewritten to the canonical call text."""
    out = ep.model_copy(deep=True)
    for step in out.steps:
        step.raw_model_text = format_tool_call_text(
            ToolCall(name=step.tool_name or "", arguments=step.arguments or {}))
    return out


def filter_teacher_episodes(
    episodes: list[Episode], scenarios: dict[str, Scenario]
) -> tuple[list[Episode], dict]:
    kept, reasons = [], Counter()
    for ep in episodes:
        why = rejection_reason(ep, scenarios[ep.scenario_id])
        if why:
            reasons[why] += 1
        else:
            kept.append(canonicalize(ep))
    return kept, {"total": len(episodes), "kept": len(kept), "rejected": dict(reasons)}


def teacher_rows(episodes: list[Episode], scenarios: dict[str, Scenario]) -> list[dict]:
    return [episode_row(e, scenarios[e.scenario_id], source="teacher") for e in episodes]


app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: Annotated[str, typer.Option(help="LiteLLM model of the teacher")],
    n: Annotated[int, typer.Option(help="how many extra scenarios to run")] = 300,
    seed: Annotated[int, typer.Option()] = 5678,
    max_new_tokens: Annotated[int, typer.Option()] = 1024,
    rpm: Annotated[int, typer.Option()] = 60,
    data: Annotated[Path, typer.Option()] = DATA,
) -> None:
    from tinyops.policies.api_policy import APIPolicy

    path = data / "train_extra" / "scenarios.jsonl"
    scenarios = load_scenarios("train_extra", data) if path.exists() else build_extra_train(seed, data)
    scenarios = scenarios[:n]
    raw = data / "train_extra" / "teacher_raw_episodes.jsonl"
    run_batched(scenarios, APIPolicy(model, max_tokens=max_new_tokens, rpm=rpm), raw, batch_size=1)
    by_id = {s.id: s for s in scenarios}
    eps = [Episode.model_validate(r) for r in read_jsonl(raw)]
    kept, stats = filter_teacher_episodes(eps, by_id)
    write_jsonl(data / "train_extra" / "teacher_episodes.jsonl", (e.model_dump(mode="json") for e in kept))
    write_json(data / "train_extra" / "teacher_filter_stats.json", {**stats, "model": model, "seed": seed})
    log.info("teacher episodes kept: %d/%d %s", stats["kept"], stats["total"], stats["rejected"])


if __name__ == "__main__":
    app()
