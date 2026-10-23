import random

import pytest

from tinyops.policies.hf_policy import HFPolicy, split_adapters
from tinyops.policies.scripted import OraclePolicy
from tinyops.scenarios.build_dataset import load_scenarios
from tinyops.sim.env import run_episode
from tinyops.train.dpo import PRIORITY, build_dpo_pairs, failure_kind, first_divergence_pair
from tinyops.train.grpo import peek_rewards
from tinyops.train.rl_data import (
    build_rl_scenarios,
    rl_dataset,
    rl_prompt_rows,
    rl_slots,
    type_weights_from_metrics,
)
from tinyops.train.sft import load_config
from tinyops.train.stage_b import choose_final, go_no_go, select_vs_reference
from tinyops.utils.io import append_jsonl


def row(step, success, unsafe, fmt=1.0, red=0.0):
    return {"step": step, "checkpoint": f"c{step}", "diagnosis_success": success, "unsafe_rate": unsafe,
            "format_valid_rate": fmt, "redundant_calls_per_ep": red, "n_episodes": 90}


def metrics(success, unsafe=0.0, fmt=1.0, red=0.0):
    def s(v):
        return {"mean": v, "lo": v, "hi": v, "n": 100}
    return {"headline": {"diagnosis_success": s(success), "unsafe_rate": s(unsafe),
                         "format_valid_rate": s(fmt), "redundant_calls_per_ep": s(red)}}


# -- selection / decisions ---------------------------------------------------------------------
def test_select_vs_reference_applies_unsafe_and_format_constraints():
    ref = row(0, 0.70, 0.04, fmt=0.98)
    curve = [row(50, 0.80, 0.10), row(100, 0.75, 0.04, fmt=0.96), row(150, 0.74, 0.02, fmt=0.975),
             row(200, 0.72, 0.0, fmt=0.99)]
    sel = select_vs_reference(curve, ref)
    assert sel["chosen"]["step"] == 150  # 50 unsafe, 100 loses > 1pt of format
    assert not sel["keep_reference"]


def test_select_vs_reference_keeps_sft_when_nothing_qualifies():
    sel = select_vs_reference([row(50, 0.9, 0.2), row(100, 0.9, 0.3)], row(0, 0.7, 0.0))
    assert sel["chosen"] is None and sel["keep_reference"]


def test_go_no_go_rules():
    sft = metrics(0.70, 0.04, red=0.30)
    assert go_no_go(metrics(0.75, 0.04, red=0.30), sft)["working"]
    matched = go_no_go(metrics(0.69, 0.02, red=0.10), sft)
    assert matched["working"] and "fewer redundant" in matched["reason"]
    assert not go_no_go(metrics(0.69, 0.02, red=0.30), sft)["working"]       # matches but not leaner
    assert not go_no_go(metrics(0.60, 0.02, red=0.0), sft)["working"]        # more than 2 pts worse
    unsafe = go_no_go(metrics(0.90, 0.08), sft)
    assert not unsafe["working"] and "safety" in unsafe["reason"]
    assert not go_no_go(metrics(0.90, 0.0, fmt=0.90), metrics(0.70, 0.04, fmt=1.0))["working"]
    assert go_no_go(metrics(0.6, 0.0), sft)["decision"] == "fallback_dpo_or_keep_sft"


def test_choose_final_prefers_sft_on_ties_and_never_picks_less_safe():
    c = {"sft": metrics(0.70, 0.04), "sft+grpo": metrics(0.70, 0.04), "sft+dpo": metrics(0.80, 0.10)}
    out = choose_final(c)
    assert out["final"] == "sft" and "sft+dpo" not in out["eligible"]
    c["sft+grpo"] = metrics(0.72, 0.03)
    assert choose_final(c)["final"] == "sft+grpo"


# -- chained adapters ----------------------------------------------------------------------------
def test_split_adapters():
    assert split_adapters("a/b") == ["a/b"]
    assert split_adapters("runs/sft/final+runs/grpo/final") == ["runs/sft/final", "runs/grpo/final"]
    with pytest.raises(ValueError):
        split_adapters(" + ")
    p = HFPolicy("m/base", adapter="me/sft+me/grpo")
    assert p.name == "base-grpo-zeroshot" and p.config()["adapter"] == "me/sft+me/grpo"


# -- RL data ---------------------------------------------------------------------------------------
def test_rl_slots_match_the_stage_b_mix():
    slots = rl_slots(800, 9012)
    assert len(slots) == 800
    diff = {d: sum(s["difficulty"] == d for s in slots) for d in ("easy", "medium", "hard")}
    assert diff == {"easy": 240, "medium": 320, "hard": 240}  # 30/40/30
    inj = [s for s in slots if s["family"]]
    assert len(inj) == 160 and sum(s["family"] == "benign_lookalike" for s in inj) == 24
    assert rl_slots(800, 9012) == slots and rl_slots(800, 1) != slots


def test_rl_slots_oversample_weak_failure_types():
    w = {"oom_killed": 3.0}
    from tinyops.scenarios.failure_types import FAILURE_TYPE_NAMES

    weights = {ft: w.get(ft, 1.0) for ft in FAILURE_TYPE_NAMES}
    slots = rl_slots(500, 1, weights)
    counts = {ft: sum(s["failure_type"] == ft for s in slots) for ft in FAILURE_TYPE_NAMES}
    assert counts["oom_killed"] > 2 * counts["crashloop_missing_env"] and sum(counts.values()) == 500


def test_type_weights_from_sft_metrics():
    groups = {"failure_type": {"oom_killed": {"headline": {"diagnosis_success": {"mean": 0.2}}},
                               "crashloop_missing_env": {"headline": {"diagnosis_success": {"mean": 1.0}}}}}
    w = type_weights_from_metrics({"groups": groups})
    assert w["oom_killed"] == pytest.approx(2.6) and w["crashloop_missing_env"] == 1.0
    assert w["hung_process_restart"] == 1.0  # missing -> neutral
    assert set(type_weights_from_metrics(None).values()) == {1.0}


def test_rl_scenarios_are_train_only_unique_and_unseen(built_data, tmp_path):
    scs = build_rl_scenarios(60, 9012, data=tmp_path)
    assert len(scs) == 60 and len({s.alert_text for s in scs}) == 60
    assert all(s.id.startswith("train_rl-") and s.split == "train" for s in scs)
    others = {s.state.model_dump_json() for sp in ("train", "val", "test", "test_injection", "test_ood")
              for s in load_scenarios(sp, built_data)}
    assert not {s.state.model_dump_json() for s in scs} & others
    assert (tmp_path / "train_rl" / "scenarios.jsonl").exists()
    rows = rl_prompt_rows(scs)
    assert rows[0]["prompt"] == [{"role": "user", "content": scs[0].alert_text}]
    assert set(rows[0]) == {"prompt", "scenario_id", "failure_type", "injected"}
    ds = rl_dataset(scs)
    assert len(ds) == 60 and ds.column_names == ["prompt", "scenario_id", "failure_type", "injected"]


# -- DPO pairs -----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def train_pairs_setup(built_data):
    scs = {s.id: s for s in load_scenarios("train", built_data)[:80]}
    gold = {i: run_episode(s.state, OraclePolicy(s.state, random.Random(1)), s.alert_text,
                           scenario_id=i) for i, s in scs.items()}
    return scs, gold


class Bad:
    name = "bad"

    def __init__(self, texts):
        self.texts = list(texts)

    def act(self, messages):
        from tinyops.policies.base import PolicyOutput
        from tinyops.policies.parsing import parse_tool_call

        t = self.texts.pop(0) if self.texts else "no idea"
        return PolicyOutput(text=t, tool_call=parse_tool_call(t).call)


def test_pair_is_built_at_the_first_divergence_with_identical_prefix(train_pairs_setup):
    scs, gold = train_pairs_setup
    sid = next(i for i, g in gold.items() if len(g.steps) >= 3)
    g = gold[sid]
    sc = scs[sid]
    # policy repeats gold's first turn, then answers in prose
    pol = run_episode(sc.state, Bad([g.steps[0].raw_model_text, "Let me think."]), sc.alert_text,
                      scenario_id=sid)
    pair = first_divergence_pair(pol, g, sc)
    assert pair["chosen"][0]["content"] == g.steps[1].raw_model_text
    assert pair["rejected"][0]["content"] == "Let me think."
    roles = [m["role"] for m in pair["prompt"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert pair["prompt"][2]["content"] == g.steps[0].raw_model_text
    assert pair["prompt"][3]["content"].startswith("Tool result for ")
    same = run_episode(sc.state, Bad([s.raw_model_text for s in g.steps]), sc.alert_text, scenario_id=sid)
    assert first_divergence_pair(same, g, sc) is None


def test_failure_kinds_and_priority(train_pairs_setup):
    scs, gold = train_pairs_setup
    sid = next(iter(gold))
    sc, g = scs[sid], gold[sid]
    prose = run_episode(sc.state, Bad(["hello"] * 3), sc.alert_text, scenario_id=sid)
    assert failure_kind(prose, sc) == "malformed"
    assert failure_kind(g, sc) is None
    events = [f'[{{"tool_name": "get_events", "parameters": {{"namespace": "{n}"}}}}]' for n in "abcdefghij"]
    loop = run_episode(sc.state, Bad(events), sc.alert_text, scenario_id=sid)
    assert failure_kind(loop, sc) == "loop"
    assert PRIORITY["obeyed_injection"] < PRIORITY["loop"] < PRIORITY["malformed"] < PRIORITY["wrong_answer"]


def test_build_pairs_prioritises_failures_and_skips_successes_and_non_train(train_pairs_setup, built_data):
    scs, gold = train_pairs_setup
    ids = list(gold)[:6]
    eps = [run_episode(scs[i].state, Bad(["hello"] * 3), scs[i].alert_text, scenario_id=i) for i in ids[:4]]
    eps.append(gold[ids[4]])  # a success: no pair
    pairs = build_dpo_pairs(eps, gold, scs, max_pairs=10, seed=1)
    assert len(pairs) == 4 and {p["kind"] for p in pairs} == {"malformed"}
    assert all(p["rejected"][0]["content"] == "hello" for p in pairs)
    assert len(build_dpo_pairs(eps, gold, scs, max_pairs=2, seed=1)) == 2
    val = load_scenarios("val", built_data)[0]
    val_ep = run_episode(val.state, Bad(["hello"] * 3), val.alert_text, scenario_id=val.id)
    assert build_dpo_pairs([val_ep], gold, {**scs, val.id: val}) == []  # never val/test


# -- GRPO helpers ----------------------------------------------------------------------------------
def test_peek_rewards_prints_real_rows(tmp_path):
    log = tmp_path / "r.jsonl"
    assert peek_rewards(log) == []
    for i in range(30):
        append_jsonl(log, {"scenario_id": f"s{i}", "total": 0.5, "diagnosis": 1.0, "format": 0.2,
                           "schema": 0.0, "efficiency": 0.1, "grounding": 0.3, "safety": 0.0,
                           "texts": ["[{\"tool_name\": \"list_pods\"}]", "Done."]})
    lines = peek_rewards(log, 10, seed=3)
    assert len(lines) == 10 and "diagnosis=+1.00" in lines[0] and "Done." in lines[0]
    assert lines == peek_rewards(log, 10, seed=3)


def test_grpo_and_dpo_configs_match_the_plan():
    g = load_config("configs/grpo.yaml")
    assert (g["lr"], g["num_generations"], g["max_completion_length"], g["batch"], g["grad_accum"],
            g["max_steps"], g["temperature"]) == (5.0e-6, 4, 384, 2, 4, 300, 0.9)
    assert g["reward_weights"] == {"diagnosis": 1.0, "format": 0.5, "schema": 0.5,
                                   "efficiency": 0.3, "grounding": 0.3, "safety": 1.0}
    assert (g["eval_every_steps"], g["val_subset"], g["seed"], g["max_steps_per_turn"]) == (50, 90, 1234, 8)
    assert 600 <= g["n_prompts"] <= 1000 and (g["batch"] * g["grad_accum"]) % g["num_generations"] == 0
    d = load_config("configs/dpo.yaml")
    assert (d["beta"], d["lr"]) == (0.1, 5.0e-6) and 1 <= d["epochs"] <= 3


def test_stage_b_cli_writes_decision_files(tmp_path):
    import subprocess
    import sys

    from tinyops.utils.io import read_json, write_json

    write_json(tmp_path / "sft.json", metrics(0.70, 0.04, red=0.3))
    write_json(tmp_path / "grpo.json", metrics(0.75, 0.02, red=0.1))
    run = lambda *a: subprocess.run([sys.executable, "-m", "tinyops.train.stage_b", *a],
                                    capture_output=True, text=True, check=True).stdout
    out = run("go-no-go", "--stage-b", str(tmp_path / "grpo.json"), "--sft", str(tmp_path / "sft.json"),
              "--out", str(tmp_path / "d.json"))
    assert "keep_stage_b" in out and read_json(tmp_path / "d.json")["working"] is True
    out = run("final", "--candidate", f"sft={tmp_path / 'sft.json'}", "--candidate",
              f"sft+grpo={tmp_path / 'grpo.json'}", "--out", str(tmp_path / "f.json"))
    assert "sft+grpo" in out and read_json(tmp_path / "f.json")["final"] == "sft+grpo"
