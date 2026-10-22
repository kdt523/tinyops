"""Run AuditKIT's agent_eval on an episodes file and write the TinyScore cross-check table.

    python -m tinyops.eval.crosscheck --episodes results/episodes/oracle_val.jsonl --split val \
        --run oracle_val
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tinyops.eval.score_episodes import load_split_scenarios
from tinyops.scoring.auditkit_bridge import (
    crosscheck,
    crosscheck_markdown,
    export_for_auditkit,
    run_auditkit_agent_eval,
    tinyscore_reference_values,
)
from tinyops.scoring.metrics import score_episode
from tinyops.sim.env import Episode
from tinyops.utils.io import read_jsonl, write_json
from tinyops.utils.log import get_logger
from tinyops.utils.paths import DATA, RESULTS

log = get_logger(__name__)


def run_crosscheck(episodes_path: Path, split: str, run: str, data: Path, out_dir: Path) -> list[dict]:
    scenarios = load_split_scenarios(split, data)
    episodes = [Episode.model_validate(r) for r in read_jsonl(episodes_path)]
    scores = [score_episode(e, scenarios[e.scenario_id]) for e in episodes]
    traj = out_dir / f"{run}.trajectories.jsonl"
    export_for_auditkit(episodes, traj)
    report = run_auditkit_agent_eval(traj, scenarios, out_dir / f"{run}.json", name=run)
    rows = crosscheck(tinyscore_reference_values(episodes, scores), report)
    write_json(out_dir / f"{run}.crosscheck.json", rows)
    (out_dir / f"{run}.crosscheck.md").write_text(
        f"### TinyScore vs AuditKIT: {run} on {split} (n={len(episodes)})\n\n"
        + crosscheck_markdown(rows), encoding="utf-8")
    flagged = [r["metric"] for r in rows if r["flag"]]
    if flagged:
        log.warning("AuditKIT and TinyScore disagree by more than 5 points on: %s", flagged)
    return rows


def main(
    episodes: Annotated[Path, typer.Option(help="episodes .jsonl")],
    split: Annotated[str, typer.Option()],
    run: Annotated[str, typer.Option(help="run name used for the output files")],
    data: Annotated[Path, typer.Option()] = DATA,
    out_dir: Annotated[Path, typer.Option()] = RESULTS / "auditkit",
) -> None:
    run_crosscheck(episodes, split, run, data, out_dir)
    log.info("wrote %s", out_dir / f"{run}.crosscheck.md")


if __name__ == "__main__":
    typer.run(main)
