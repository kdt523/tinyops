"""Live mode for the Space: run KubeSim + a model step by step (needs a GPU and the model weights).

Kept in the package so it is unit-tested with a stub policy; the Space only wires it to the UI.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

from tinyops.scenarios.base import Scenario
from tinyops.scenarios.build_dataset import load_scenarios, sub_seed
from tinyops.scenarios.injection import ATTACK_FAMILIES, feasible_locations, inject
from tinyops.scenarios.names import TRAIN_POOLS
from tinyops.sim.env import Episode, EpisodeSession
from tinyops.utils.paths import DATA


def pick_scenario(seed: int, inject_attack: bool, data: Path = DATA, split: str = "val") -> Scenario:
    """A random clean val scenario, optionally with a (seeded) injected attack planted in it."""
    rng = random.Random(sub_seed(seed, "live-pick"))
    clean = [s for s in load_scenarios(split, data) if not s.injected]
    sc = rng.choice(clean)
    if not inject_attack:
        return sc
    state = sc.state
    family = rng.choice(ATTACK_FAMILIES)
    location = rng.choice(feasible_locations(state))
    state = inject(state, family, location, rng, TRAIN_POOLS)
    return sc.model_copy(update={"state": state, "injected": True, "injection_family": family,
                                 "injection_location": location, "id": f"{sc.id}+{family}"})


def live_steps(sc: Scenario, policy, max_steps: int = 8) -> Iterator[dict]:
    """Yield one dict per step: {"step": StepRecord-as-dict, "done": bool}; the last carries the Episode."""
    session = EpisodeSession(sc.state, sc.alert_text, getattr(policy, "name", "live"), max_steps,
                             scenario_id=sc.id)
    while not session.done:
        try:
            out = policy.act(session.messages)
        except Exception as e:  # noqa: BLE001 - show the failure in the UI instead of crashing it
            session.fail(e)
            break
        rec = session.apply(out)
        yield {"step": rec.model_dump(), "done": session.done}
    ep: Episode = session.episode()
    yield {"step": None, "done": True, "episode": ep.model_dump(mode="json")}
