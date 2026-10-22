"""Few-shot examples: oracle episodes from the TRAIN split only (never val/test)."""

from __future__ import annotations

import random
from pathlib import Path

from tinyops.policies.scripted import OraclePolicy
from tinyops.scenarios.build_dataset import load_scenarios, sub_seed
from tinyops.sim.env import Episode, run_episode
from tinyops.utils.paths import DATA


def build_fewshot_episodes(k: int = 2, seed: int = 1234, data: Path = DATA) -> list[Episode]:
    """k oracle episodes of k different failure types, non-injected, from the train split."""
    if k < 1:
        return []
    train = [s for s in load_scenarios("train", data) if not s.injected]
    assert all(s.split == "train" for s in train)
    rng = random.Random(sub_seed(seed, "fewshot"))
    rng.shuffle(train)
    chosen, types = [], set()
    for sc in train:
        if sc.failure_type in types:
            continue
        types.add(sc.failure_type)
        chosen.append(sc)
        if len(chosen) == k:
            break
    if len(chosen) < k:
        raise ValueError(f"cannot pick {k} few-shot examples of different failure types")
    return [
        run_episode(sc.state, OraclePolicy(sc.state, random.Random(sub_seed(seed, f"shot-{sc.id}"))),
                    sc.alert_text, seed=seed, scenario_id=sc.id)
        for sc in chosen
    ]
