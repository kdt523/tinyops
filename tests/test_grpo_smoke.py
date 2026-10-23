"""Plumbing smoke test for GRPO (slow; run with `pytest -m slow tests/test_grpo_smoke.py`).

SmolLM2-135M, 3 optimizer steps, 2 generations: checks that tools hit the right scenario, rewards
are called with the expected inputs, and trajectories/logs/selection files are written.
"""

import json

import pytest

from tinyops.scoring import rewards as rewards_mod
from tinyops.train import grpo as grpo_mod
from tinyops.train.sft import load_config
from tinyops.utils.io import read_jsonl


@pytest.mark.slow
def test_grpo_plumbing_on_a_tiny_model(built_data, tmp_path, monkeypatch):
    pytest.importorskip("agenttune")
    seen_kwargs: list[dict] = []

    class SpyRewards(rewards_mod.RewardSet):
        def breakdown(self, kwargs):
            seen_kwargs.append({k: (len(v) if hasattr(v, "__len__") else v) for k, v in kwargs.items()
                                if k in ("scenario_id", "trajectories", "conversations", "prompts")})
            return super().breakdown(kwargs)

    monkeypatch.setattr(grpo_mod, "RewardSet", SpyRewards)
    cfg = load_config("configs/grpo.yaml", {
        "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct", "start_from": None, "n_prompts": 6,
        "max_steps": 3, "num_generations": 2, "batch": 2, "grad_accum": 1, "eval_every_steps": 3,
        "val_subset": 9, "eval_batch_size": 9, "max_completion_length": 96, "max_steps_per_turn": 3,
        "peek_every_steps": 1, "data_dir": str(built_data), "output_dir": str(tmp_path / "grpo"),
        "results_dir": str(tmp_path / "res"), "sft_val_metrics": None, "sft_selection": None,
        "push_to_hub": None,
    })
    sel = grpo_mod.run_grpo(cfg)

    run = json.loads((tmp_path / "res" / "runs" / "grpo_train_run.json").read_text())
    assert run["rollouts_bound"] >= 3 and run["unbound_rollouts"] == 0  # every rollout found its scenario
    assert run["beta"] is not None
    assert seen_kwargs and all(k["scenario_id"] == k["trajectories"] for k in seen_kwargs)

    rows = read_jsonl(tmp_path / "res" / "train_logs" / "grpo_rewards.jsonl")
    prompts = {json.loads(line)["id"] for line in
               (built_data / "train_rl" / "scenarios.jsonl").read_text().splitlines()}
    assert rows and {r["scenario_id"] for r in rows} <= prompts
    assert all(set(r) >= {"diagnosis", "format", "schema", "efficiency", "grounding", "safety", "total"}
               for r in rows)
    assert (tmp_path / "grpo" / "trajectories.jsonl").exists()
    assert (tmp_path / "res" / "train_logs" / "grpo_val_curve.csv").exists()
    assert (tmp_path / "res" / "runs" / "grpo_selection.json").exists()
    assert "keep_reference" in sel
