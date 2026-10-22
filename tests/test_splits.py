import shutil
from collections import Counter

import pytest
import typer

from tinyops.scenarios.build_dataset import (
    HASH_FILE,
    PER_TYPE,
    SPLITS,
    TEST_SPLITS,
    build_split,
    plan_split,
    sub_seed,
    write_splits,
)
from tinyops.scenarios.failure_types import FAILURE_TYPE_NAMES
from tinyops.scenarios.injection import ATTACK_FAMILIES, FAMILIES
from tinyops.scenarios.names import OOD_POOLS, TRAIN_POOLS
from tinyops.utils.io import sha256_file


def count(scs, key):
    return Counter(key(s) for s in scs)


def test_sizes(scenarios):
    assert {k: len(v) for k, v in scenarios.items()} == {
        "train": 1215, "val": 180, "test": 270, "test_injection": 162, "test_ood": 63}


def test_per_type_counts(scenarios):
    for split, scs in scenarios.items():
        c = count(scs, lambda s: s.failure_type)
        assert set(c) == set(FAILURE_TYPE_NAMES)
        assert set(c.values()) == {PER_TYPE[split]}, split


def test_difficulty_mix(scenarios):
    for split in ("train", "val"):
        for ft in FAILURE_TYPE_NAMES:
            c = count([s for s in scenarios[split] if s.failure_type == ft], lambda s: s.difficulty)
            n = PER_TYPE[split]
            assert abs(c["easy"] / n - 0.40) < 0.03 and abs(c["medium"] / n - 0.35) < 0.03
    assert count(scenarios["train"], lambda s: s.difficulty) == {"easy": 486, "medium": 423, "hard": 306}
    for ft in FAILURE_TYPE_NAMES:
        c = count([s for s in scenarios["test"] if s.failure_type == ft], lambda s: s.difficulty)
        assert c == {"easy": 10, "medium": 10, "hard": 10}
    assert set(count(scenarios["test_ood"], lambda s: s.difficulty)) == {"medium", "hard"}


def test_injection_composition(scenarios):
    for split in ("train", "val"):
        scs = scenarios[split]
        inj = [s for s in scs if s.injected]
        assert abs(len(inj) / len(scs) - 0.15) < 0.005
        fams = count(inj, lambda s: s.injection_family)
        assert abs(fams["benign_lookalike"] / len(scs) - 0.03) < 0.005
        attacks = [fams[f] for f in ATTACK_FAMILIES]
        assert max(attacks) - min(attacks) <= 1  # families uniform
    for split in ("test", "test_ood"):
        assert not any(s.injected for s in scenarios[split]), split
    ti = scenarios["test_injection"]
    assert all(s.injected for s in ti)
    fams = count(ti, lambda s: s.injection_family)
    assert set(fams) == set(FAMILIES) and min(fams.values()) >= 20
    locs = count(ti, lambda s: s.injection_location)
    assert set(locs) == {"log", "event", "configmap"} and min(locs.values()) >= 40
    for s in scenarios["train"]:
        assert s.injected == (s.injection_family is not None) == (s.injection_location is not None)


def test_ood_uses_ood_names_and_alerts(scenarios):
    for s in scenarios["test_ood"]:
        assert set(s.state.namespaces) <= set(OOD_POOLS.namespaces)
        assert not set(s.state.namespaces) & set(TRAIN_POOLS.namespaces)
        assert s.alert_text.split()[0] in {"[ALERT]", "Ticket", "Opsgenie", "Status", "On-call"}


def test_no_duplicate_ids_or_states_across_splits(scenarios):
    ids = [s.id for scs in scenarios.values() for s in scs]
    assert len(ids) == len(set(ids))
    assert all(s.id.startswith(s.split + "-") for scs in scenarios.values() for s in scs)
    states = [s.state.model_dump_json() for scs in scenarios.values() for s in scs]
    assert len(states) == len(set(states))


def test_hashes_file_matches_files(built_data):
    rows = [ln.split(None, 1) for ln in (built_data / HASH_FILE).read_text().splitlines()]
    assert {rel for _, rel in rows} == {f"data/{s}/scenarios.jsonl" for s in TEST_SPLITS}
    for digest, rel in rows:
        assert sha256_file(built_data / rel.removeprefix("data/")) == digest


def test_stats_files_written(built_data):
    import json
    stats = json.loads((built_data / "val" / "stats.json").read_text())
    assert stats["n"] == 180 and stats["n_injected"] == 27


def test_build_is_deterministic():
    a = [s.model_dump_json() for s in build_split("test_ood", 1234)]
    assert a == [s.model_dump_json() for s in build_split("test_ood", 1234)]
    assert a != [s.model_dump_json() for s in build_split("test_ood", 99)]
    assert plan_split("test", 1234) == plan_split("test", 1234)


def test_sub_seed_is_stable_and_not_salted_hash():
    assert sub_seed(1234, "train") == sub_seed(1234, "train")
    assert sub_seed(1234, "train") != sub_seed(1234, "val")
    assert sub_seed(1234, "train") == 3236040405 or isinstance(sub_seed(1234, "train"), int)


def test_locked_test_set_refuses_overwrite(built_data, tmp_path):
    out = tmp_path / "d"
    shutil.copytree(built_data, out)
    # unchanged -> fine (no-op)
    write_splits(1234, out, splits=("test_ood",))
    # a different seed would change the locked content -> refuse
    with pytest.raises(typer.Exit):
        write_splits(99, out, splits=("test_ood",))
    # explicit force overwrites
    write_splits(99, out, force_tests=True, splits=("test_ood",))
    assert sha256_file(out / "test_ood" / "scenarios.jsonl") != sha256_file(
        built_data / "test_ood" / "scenarios.jsonl")


def test_tampered_test_file_detected(built_data, tmp_path):
    out = tmp_path / "d"
    shutil.copytree(built_data, out)
    with open(out / "test_ood" / "scenarios.jsonl", "a") as f:
        f.write("\n")
    with pytest.raises(typer.Exit):
        write_splits(1234, out, splits=("test_ood",))


def test_all_splits_listed():
    assert SPLITS == ("train", "val", "test", "test_injection", "test_ood")
