"""Scenarios and prompt dataset for Stage B (GRPO / DPO). Train distribution only, never eval data.

`data/train_rl` is generated with its own seed (default 9012): difficulty mix 30/40/30, 20%
injected (3% harmless lookalikes), failure types weighted toward where SFT is weakest. Alert texts
are unique, because a rollout's scenario is identified from the alert in its prompt.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from tinyops.scenarios.base import Scenario
from tinyops.scenarios.build_dataset import build_from_slots, sub_seed
from tinyops.scenarios.failure_types import FAILURE_TYPE_NAMES
from tinyops.scenarios.injection import ATTACK_FAMILIES
from tinyops.utils.io import read_json, write_jsonl
from tinyops.utils.paths import DATA

DIFFICULTY_MIX = {"easy": 0.30, "medium": 0.40, "hard": 0.30}


def type_weights_from_metrics(metrics: dict | None, strength: float = 2.0) -> dict[str, float]:
    """Weight each failure type by how badly SFT does on it (1 + strength * (1 - success))."""
    if not metrics:
        return {ft: 1.0 for ft in FAILURE_TYPE_NAMES}
    groups = metrics["groups"]["failure_type"]
    return {ft: 1.0 + strength * (1.0 - (groups[ft]["headline"]["diagnosis_success"]["mean"] or 0.0))
            if ft in groups else 1.0 for ft in FAILURE_TYPE_NAMES}


def _apportion(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Largest-remainder apportionment of `total` slots over the weights."""
    s = sum(weights.values())
    raw = {k: total * w / s for k, w in weights.items()}
    out = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


def rl_slots(n: int, seed: int, type_weights: dict[str, float] | None = None,
             injected_share: float = 0.20, benign_share: float = 0.03) -> list[dict]:
    rng = random.Random(sub_seed(seed, "rl-plan"))
    counts = _apportion(n, type_weights or {ft: 1.0 for ft in FAILURE_TYPE_NAMES})
    diffs = [d for d, c in _apportion(n, DIFFICULTY_MIX).items() for _ in range(c)]
    rng.shuffle(diffs)
    types = [ft for ft, c in counts.items() for _ in range(c)]
    rng.shuffle(types)
    slots = [{"failure_type": t, "difficulty": d, "family": None} for t, d in zip(types, diffs, strict=True)]
    n_inj, n_benign = round(injected_share * n), round(benign_share * n)
    for j, i in enumerate(rng.sample(range(n), n_inj)):
        slots[i]["family"] = ("benign_lookalike" if j < n_benign
                              else ATTACK_FAMILIES[(j - n_benign) % len(ATTACK_FAMILIES)])
    return slots


def build_rl_scenarios(n: int = 800, seed: int = 9012, type_weights: dict[str, float] | None = None,
                       data: Path | None = DATA) -> list[Scenario]:
    """`n` scenarios with unique alert texts (extra slots are generated and de-duplicated)."""
    slots = rl_slots(int(n * 1.5), seed, type_weights)
    scenarios = build_from_slots("train", slots, seed, name="train_rl")
    seen, kept = set(), []
    for sc in scenarios:
        if sc.alert_text not in seen:
            seen.add(sc.alert_text)
            kept.append(sc)
    if len(kept) < n:
        raise ValueError(f"only {len(kept)} unique-alert scenarios from {len(scenarios)}; raise the pool")
    kept = kept[:n]
    if data is not None:
        write_jsonl(Path(data) / "train_rl" / "scenarios.jsonl", (s.model_dump(mode="json") for s in kept))
    return kept


def rl_prompt_rows(scenarios: list[Scenario]) -> list[dict]:
    """Dataset rows: the prompt is only the alert (AgentTune adds the system prompt and tools)."""
    return [{"prompt": [{"role": "user", "content": s.alert_text}], "scenario_id": s.id,
             "failure_type": s.failure_type, "injected": s.injected} for s in scenarios]


def rl_dataset(scenarios: list[Scenario]):
    from datasets import Dataset

    return Dataset.from_list(rl_prompt_rows(scenarios))


def load_type_weights(path: str | Path | None) -> dict[str, float]:
    return type_weights_from_metrics(read_json(path) if path and Path(path).exists() else None)
