"""Run a policy over a split and append one Episode per scenario to a jsonl file (resumable).

Only the scripted `oracle` and `random` policies for now; skill 5 adds model policies.

    python -m tinyops.eval.run_eval --policy oracle --split val --out results/episodes/oracle_val.jsonl
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from tinyops.policies.random_policy import RandomPolicy
from tinyops.policies.scripted import OraclePolicy
from tinyops.scenarios.base import Scenario
from tinyops.scenarios.build_dataset import TEST_SPLITS, load_scenarios, sub_seed
from tinyops.sim.env import run_episode
from tinyops.utils.io import append_jsonl, read_jsonl
from tinyops.utils.log import get_logger
from tinyops.utils.paths import DATA

log = get_logger(__name__)

PolicyFactory = Callable[[Scenario, int], object]
POLICIES: dict[str, PolicyFactory] = {
    "oracle": lambda sc, seed: OraclePolicy(
        sc.state, random.Random(sub_seed(seed, f"oracle-{sc.id}"))),
    "random": lambda sc, seed: RandomPolicy(random.Random(sub_seed(seed, f"random-{sc.id}"))),
}


def run_policy(
    scenarios: list[Scenario], factory: PolicyFactory, out: Path, seed: int = 1234,
    max_steps: int = 8,
) -> int:
    """Append episodes for scenarios not already in `out`. Returns the number newly run."""
    done = {r["scenario_id"] for r in read_jsonl(out)} if out.exists() else set()
    n = 0
    for sc in scenarios:
        if sc.id in done:
            continue
        ep = run_episode(sc.state, factory(sc, seed), sc.alert_text, max_steps=max_steps,
                         seed=seed, scenario_id=sc.id)
        append_jsonl(out, ep.model_dump(mode="json"))
        n += 1
    return n


def main(
    policy: Annotated[str, typer.Option(help="oracle | random")],
    split: Annotated[str, typer.Option(help="train | val | test | test_injection | test_ood")],
    out: Annotated[Path, typer.Option(help="episodes .jsonl (appended; resumable)")],
    seed: Annotated[int, typer.Option()] = 1234,
    limit: Annotated[int, typer.Option(help="0 = all scenarios")] = 0,
    max_steps: Annotated[int, typer.Option()] = 8,
    data: Annotated[Path, typer.Option()] = DATA,
) -> None:
    if policy not in POLICIES:
        raise typer.BadParameter(f"unknown policy {policy!r}; choose from {sorted(POLICIES)}")
    if split in TEST_SPLITS:
        log.warning("running %s on locked test split %s: base/scripted policies only, never tune on it",
                    policy, split)
    scenarios = load_scenarios(split, data)
    if limit:
        scenarios = scenarios[:limit]
    n = run_policy(scenarios, POLICIES[policy], out, seed, max_steps)
    log.info("ran %d new episodes (%d total requested) -> %s", n, len(scenarios), out)


if __name__ == "__main__":
    typer.run(main)
