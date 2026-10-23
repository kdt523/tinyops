"""Build the locked splits and export gold trajectories.

    python -m tinyops.scenarios.build_dataset --seed 1234 --out data/
    python -m tinyops.scenarios.build_dataset gold --split all --seed 1234
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from tinyops.policies.scripted import OraclePolicy
from tinyops.scenarios.alerts import make_alert
from tinyops.scenarios.base import Scenario
from tinyops.scenarios.failure_types import FAILURE_TYPE_NAMES, FAILURE_TYPES
from tinyops.scenarios.injection import ATTACK_FAMILIES, FAMILIES, feasible_locations, inject
from tinyops.scenarios.names import OOD_POOLS, TRAIN_POOLS
from tinyops.sim.env import run_episode
from tinyops.sim.export import write_trajectories
from tinyops.utils.io import read_jsonl, sha256_file, write_json, write_jsonl
from tinyops.utils.log import get_logger
from tinyops.utils.paths import DATA

log = get_logger(__name__)

SPLITS = ("train", "val", "test", "test_injection", "test_ood")
TEST_SPLITS = ("test", "test_injection", "test_ood")
HASH_FILE = "TEST_HASHES.txt"
N = len(FAILURE_TYPE_NAMES)

# per-type counts
PER_TYPE = {"train": 135, "val": 20, "test": 30, "test_injection": 18, "test_ood": 7}
INJECTED_SHARE = {"train": 0.15, "val": 0.15}
BENIGN_SHARE = 0.03


def sub_seed(seed: int, key: str) -> int:
    """Deterministic sub-seed (never Python's salted hash)."""
    return int(hashlib.sha256(f"{seed}-{key}".encode()).hexdigest(), 16) % 2**32


def _difficulty_counts(split: str, per_type: int) -> dict[str, int]:
    if split in ("train", "val"):  # 40/35/25
        easy = round(per_type * 0.40)
        med = round(per_type * 0.35)
        return {"easy": easy, "medium": med, "hard": per_type - easy - med}
    if split == "test_ood":
        return {"easy": 0, "medium": 4, "hard": per_type - 4}
    third = per_type // 3
    return {"easy": third, "medium": third, "hard": per_type - 2 * third}


def plan_split(split: str, seed: int) -> list[dict]:
    """The ordered slots (type, difficulty, family) of a split. Pure function of (split, seed)."""
    rng = random.Random(sub_seed(seed, f"plan-{split}"))
    per_type = PER_TYPE[split]
    slots = []
    for ft in FAILURE_TYPE_NAMES:
        diffs = [d for d, c in _difficulty_counts(split, per_type).items() for _ in range(c)]
        rng.shuffle(diffs)
        slots += [{"failure_type": ft, "difficulty": d, "family": None} for d in diffs]
    rng.shuffle(slots)
    total = len(slots)
    if split == "test_injection":
        fams = [FAMILIES[i % len(FAMILIES)] for i in range(total)]
        rng.shuffle(fams)
        for slot, fam in zip(slots, fams, strict=True):
            slot["family"] = fam
    elif split in INJECTED_SHARE:
        n_inj = round(INJECTED_SHARE[split] * total)
        n_benign = round(BENIGN_SHARE * total)
        picked = rng.sample(range(total), n_inj)
        for j, i in enumerate(picked):
            slots[i]["family"] = ("benign_lookalike" if j < n_benign
                                  else ATTACK_FAMILIES[(j - n_benign) % len(ATTACK_FAMILIES)])
    return slots


def build_from_slots(
    split: str, slots: list[dict], seed: int, name: str | None = None
) -> list[Scenario]:
    """Generate scenarios for ordered slots (type, difficulty, family). Deterministic."""
    pools = OOD_POOLS if split == "test_ood" else TRAIN_POOLS
    loc_counts: Counter[str] = Counter()
    out: list[Scenario] = []
    for idx, slot in enumerate(slots):
        rng = random.Random(sub_seed(seed, f"{name or split}-{idx}"))
        ft, diff, fam = slot["failure_type"], slot["difficulty"], slot["family"]
        state = FAILURE_TYPES[ft](rng, diff, pools)
        location = None
        if fam is not None:
            feasible = feasible_locations(state)
            least = min(loc_counts[loc] for loc in feasible)
            location = rng.choice([loc for loc in feasible if loc_counts[loc] == least])
            loc_counts[location] += 1
            state = inject(state, fam, location, rng, pools)
        alert = make_alert(rng, state, ood=split == "test_ood")
        out.append(Scenario(
            id=f"{name or split}-{ft}-{idx:05d}", split=split, failure_type=ft, difficulty=diff,
            injected=fam is not None, injection_family=fam, injection_location=location,
            alert_text=alert, state=state,
        ))
    return out


def build_split(split: str, seed: int, name: str | None = None) -> list[Scenario]:
    """Scenarios of `split`. `name` only changes the id prefix (e.g. train_extra)."""
    return build_from_slots(split, plan_split(split, seed), seed, name)


def split_stats(scenarios: list[Scenario]) -> dict:
    return {
        "n": len(scenarios),
        "by_failure_type": dict(sorted(Counter(s.failure_type for s in scenarios).items())),
        "by_difficulty": dict(sorted(Counter(s.difficulty for s in scenarios).items())),
        "by_injection_family": dict(sorted(Counter(s.injection_family for s in scenarios if s.injected).items())),
        "by_injection_location": dict(sorted(Counter(s.injection_location for s in scenarios if s.injected).items())),
        "n_injected": sum(s.injected for s in scenarios),
    }


def _serialize(scenarios: list[Scenario]) -> bytes:
    lines = (json.dumps(s.model_dump(mode="json"), ensure_ascii=False) + "\n" for s in scenarios)
    return "".join(lines).encode("utf-8")


def read_hash_file(out: Path) -> dict[str, str]:
    path = out / HASH_FILE
    if not path.exists():
        return {}
    rows = [ln.split(None, 1) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return {rel.strip(): digest for digest, rel in rows}


def write_splits(
    seed: int, out: Path, force_tests: bool = False, splits: tuple[str, ...] = SPLITS
) -> dict[str, str]:
    """Generate all splits. Locked test files are never silently overwritten."""
    locked = read_hash_file(out)
    new_hashes: dict[str, str] = {}
    for split in splits:
        scenarios = build_split(split, seed)
        data = _serialize(scenarios)
        path = out / split / "scenarios.jsonl"
        rel = f"data/{split}/scenarios.jsonl"
        if split in TEST_SPLITS and path.exists() and not force_tests:
            on_disk = sha256_file(path)
            new = hashlib.sha256(data).hexdigest()
            if locked.get(rel) != on_disk:
                raise typer.Exit(_fail(f"{rel} does not match {HASH_FILE}: the locked test set "
                                       "was modified. Restore it or pass --force-regenerate-tests."))
            if new != on_disk:
                raise typer.Exit(_fail(f"regenerating {rel} would change the locked test set "
                                       "(generator code changed?). Refusing; pass "
                                       "--force-regenerate-tests only if you accept re-running "
                                       "ALL final evaluations."))
            log.info("%s is locked and unchanged", rel)
        else:
            if split in TEST_SPLITS and path.exists():
                log.warning("!!! OVERWRITING LOCKED TEST SPLIT %s (--force-regenerate-tests) !!!", rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        write_json(out / split / "stats.json", split_stats(scenarios))
    for split in TEST_SPLITS:
        path = out / split / "scenarios.jsonl"
        if path.exists():
            new_hashes[f"data/{split}/scenarios.jsonl"] = sha256_file(path)
    (out / HASH_FILE).write_text(
        "".join(f"{h}  {rel}\n" for rel, h in sorted(new_hashes.items())), encoding="utf-8")
    return new_hashes


def verify_test_hashes(data: Path = DATA) -> dict[str, str]:
    """Check data/TEST_HASHES.txt against the files on disk. Raises on any mismatch."""
    locked = read_hash_file(data)
    if not locked:
        raise FileNotFoundError(f"{data / HASH_FILE} is missing or empty")
    expected = {f"data/{s}/scenarios.jsonl" for s in TEST_SPLITS}
    if set(locked) != expected:
        raise ValueError(f"{HASH_FILE} must list exactly {sorted(expected)}, found {sorted(locked)}")
    bad = []
    for rel, digest in locked.items():
        path = data / rel.removeprefix("data/")
        if not path.exists() or sha256_file(path) != digest:
            bad.append(rel)
    if bad:
        raise ValueError(f"locked test files do not match {HASH_FILE}: {bad}")
    return locked


def _fail(msg: str) -> int:
    log.error(msg)
    return 1


def load_scenarios(split: str, data: Path = DATA) -> list[Scenario]:
    return [Scenario.model_validate(r) for r in read_jsonl(data / split / "scenarios.jsonl")]


def run_gold(split: str, seed: int, data: Path) -> None:
    scenarios = load_scenarios(split, data)
    episodes = []
    for sc in scenarios:
        rng = random.Random(sub_seed(seed, f"oracle-{sc.id}"))
        policy = OraclePolicy(sc.state, rng)
        episodes.append(run_episode(sc.state, policy, sc.alert_text, scenario_id=sc.id, seed=seed))
    write_jsonl(data / split / "gold_episodes.jsonl", (e.model_dump(mode="json") for e in episodes))
    write_trajectories(episodes, data / split / "gold_trajectories.jsonl")
    log.info("wrote %d gold episodes for %s", len(episodes), split)


app = typer.Typer(invoke_without_command=True, add_completion=False, no_args_is_help=False)


@app.callback()
def build(
    ctx: typer.Context,
    seed: Annotated[int, typer.Option(help="Master seed")] = 1234,
    out: Annotated[Path, typer.Option(help="Output data directory")] = DATA,
    force_regenerate_tests: Annotated[bool, typer.Option(
        "--force-regenerate-tests", help="Overwrite locked test splits (loud warning)")] = False,
) -> None:
    """Generate train/val/test/test_injection/test_ood and the test hash file."""
    if ctx.invoked_subcommand is not None:
        return
    hashes = write_splits(seed, out, force_regenerate_tests)
    for rel, h in sorted(hashes.items()):
        log.info("%s  %s", h, rel)


@app.command()
def gold(
    split: Annotated[str, typer.Option(help="train | val | all")] = "all",
    seed: Annotated[int, typer.Option()] = 1234,
    data: Annotated[Path, typer.Option(help="Data directory")] = DATA,
) -> None:
    """Run the oracle on train/val and write gold episodes and trajectories. Never for test."""
    if split not in ("train", "val", "all"):
        raise typer.BadParameter("gold trajectories are only produced for train and val")
    for s in ("train", "val") if split == "all" else (split,):
        run_gold(s, seed, data)


@app.command()
def verify(data: Annotated[Path, typer.Option(help="Data directory")] = DATA) -> None:
    """Verify the locked test splits against data/TEST_HASHES.txt (exit 1 on mismatch)."""
    try:
        locked = verify_test_hashes(data)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        raise typer.Exit(1) from e
    log.info("TEST_HASHES.txt verified for %d files", len(locked))


if __name__ == "__main__":
    app()
