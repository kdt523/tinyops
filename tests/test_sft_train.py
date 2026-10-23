import json
from types import SimpleNamespace

import pytest
import yaml

from tinyops.scenarios.build_dataset import load_scenarios
from tinyops.train.sft import base_unsafe_rate, load_config, write_run_provenance
from tinyops.train.val_eval import (
    append_curve,
    generation_mode,
    read_curve,
    select_checkpoint,
    select_val_subset,
    write_selection,
)


def row(step, success, unsafe, fmt=1.0, red=0.0):
    return {"step": step, "checkpoint": f"ckpt-{step}", "diagnosis_success": success,
            "unsafe_rate": unsafe, "format_valid_rate": fmt, "redundant_calls_per_ep": red,
            "n_episodes": 90}


def test_selection_prefers_success_under_the_safety_constraint():
    curve = [row(100, 0.50, 0.00), row(200, 0.80, 0.10), row(300, 0.70, 0.02), row(400, 0.60, 0.0)]
    sel = select_checkpoint(curve, base_unsafe=0.30)
    assert sel["chosen"]["step"] == 300 and sel["constraint_met"]  # 200 breaks the 5% cap
    assert sel["unsafe_threshold"] == 0.05


def test_base_unsafe_rate_tightens_the_cap():
    curve = [row(100, 0.9, 0.04), row(200, 0.6, 0.0)]
    assert select_checkpoint(curve, base_unsafe=0.04)["chosen"]["step"] == 100
    assert select_checkpoint(curve, base_unsafe=0.01)["chosen"]["step"] == 200
    assert select_checkpoint(curve, base_unsafe=None)["chosen"]["step"] == 100


def test_selection_ties_break_on_format_then_redundancy_then_earlier_step():
    curve = [row(100, 0.7, 0, fmt=0.95), row(200, 0.7, 0, fmt=0.99, red=0.5),
             row(300, 0.7, 0, fmt=0.99, red=0.1), row(400, 0.7, 0, fmt=0.99, red=0.1)]
    assert select_checkpoint(curve, None)["chosen"]["step"] == 300


def test_selection_falls_back_to_least_unsafe_and_says_so():
    sel = select_checkpoint([row(100, 0.9, 0.2), row(200, 0.5, 0.1)], base_unsafe=0.0)
    assert not sel["constraint_met"] and sel["chosen"]["step"] == 200
    with pytest.raises(ValueError):
        select_checkpoint([], None)


def test_missing_unsafe_rate_counts_as_safe():
    assert select_checkpoint([row(1, 0.4, None)], 0.0)["constraint_met"]


def test_curve_roundtrip_and_selection_file(tmp_path):
    path = tmp_path / "curve.csv"
    append_curve(path, row(100, 0.5, 0.0))
    append_curve(path, row(200, 0.75, None))
    rows = read_curve(path)
    assert rows[0]["step"] == 100 and rows[1]["unsafe_rate"] is None and rows[1]["diagnosis_success"] == 0.75
    sel = write_selection(path, tmp_path / "runs" / "sft_selection.json", base_unsafe=0.5)
    saved = json.loads((tmp_path / "runs" / "sft_selection.json").read_text())
    assert saved["chosen"]["step"] == 200 and saved["rule"].startswith("max val")
    assert sel["candidates"] == rows


def test_val_subset_is_balanced_seeded_and_includes_injected(built_data):
    val = load_scenarios("val", built_data)
    sub = select_val_subset(val, per_type=10, seed=1234)
    assert len(sub) == 90 and {s.id for s in sub} <= {s.id for s in val}
    from collections import Counter

    assert set(Counter(s.failure_type for s in sub).values()) == {10}
    assert 9 <= sum(s.injected for s in sub) <= 27
    assert [s.id for s in sub] == [s.id for s in select_val_subset(val, per_type=10, seed=1234)]
    assert [s.id for s in sub] != [s.id for s in select_val_subset(val, per_type=10, seed=7)]
    assert len(select_val_subset(val, per_type=1)) == 9


def test_generation_mode_restores_training_state():
    calls = []

    class M:
        training = True
        is_gradient_checkpointing = True
        config = SimpleNamespace(use_cache=False)

        def eval(self):
            self.training = False
            calls.append("eval")

        def train(self):
            self.training = True
            calls.append("train")

        def gradient_checkpointing_disable(self):
            calls.append("gc_off")

        def gradient_checkpointing_enable(self, **kw):
            calls.append(("gc_on", kw))

    m, tok = M(), SimpleNamespace(padding_side="right")
    with generation_mode(m, tok):
        assert not m.training and m.config.use_cache is True and tok.padding_side == "left"
    assert m.training and m.config.use_cache is False and tok.padding_side == "right"
    assert calls[:2] == ["eval", "gc_off"] and calls[-1] == "train"
    assert ("gc_on", {"gradient_checkpointing_kwargs": {"use_reentrant": False}}) in calls


def test_generation_mode_restores_even_when_generation_raises():
    m = SimpleNamespace(training=True, is_gradient_checkpointing=False,
                        config=SimpleNamespace(use_cache=False), eval=lambda: None,
                        train=lambda: None)
    tok = SimpleNamespace(padding_side="right")
    with pytest.raises(RuntimeError), generation_mode(m, tok):
        raise RuntimeError("OOM")
    assert tok.padding_side == "right" and m.config.use_cache is False


def test_config_file_matches_the_plan_and_overrides_work(tmp_path):
    cfg = load_config("configs/sft.yaml")
    assert cfg["base_model"] == "CohereLabs/tiny-aya-global" and cfg["lr"] == 1e-4
    assert cfg["lora"] == {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"}
    assert (cfg["epochs"], cfg["grad_accum"], cfg["eval_every_steps"], cfg["val_subset"]) == (2, 16, 100, 90)
    assert cfg["max_seq_len"] == 4096 and cfg["seed"] == 1234
    assert load_config("configs/sft.yaml", {"lr": 5e-5})["lr"] == 5e-5
    assert yaml.safe_load("5e-5") == 5e-5 or isinstance(yaml.safe_load("5.0e-5"), float)


def test_base_unsafe_rate_reads_the_metrics_file(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"headline": {"unsafe_rate": {"mean": 0.12}}}))
    assert base_unsafe_rate(str(p)) == 0.12
    assert base_unsafe_rate(str(tmp_path / "missing.json")) is None
    assert base_unsafe_rate(None) is None


def test_provenance_is_written_with_the_lexsi_schema(tmp_path):
    cfg = load_config("configs/sft.yaml")
    rec = write_run_provenance(tmp_path, cfg, {"examples": 10})
    saved = json.loads((tmp_path / "lexsi_provenance.json").read_text())
    assert saved["schema"] == "lexsi.provenance/1" and saved["method"] == "sft_lora"
    assert saved["base_model"] == "CohereLabs/tiny-aya-global" and saved["params"]["dataset_stats"] == {"examples": 10}
    assert rec["method"] == "sft_lora"


@pytest.mark.slow
def test_sft_end_to_end_smoke_on_a_tiny_model(built_data, tmp_path):
    """Plumbing only: SmolLM2-135M, 2 optimizer steps, LoRA, assistant-only labels, val callback."""
    from tinyops.train.sft import run_sft

    cfg = load_config("configs/sft.yaml", {
        "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct", "epochs": 1, "max_steps": 2,
        "grad_accum": 1, "eval_every_steps": 1, "val_subset": 9, "eval_batch_size": 9,
        "data_dir": str(built_data), "output_dir": str(tmp_path / "sft"),
        "results_dir": str(tmp_path / "res"), "base_metrics": None, "push_to_hub": None,
    })
    sel = run_sft(cfg)
    curve = read_curve(tmp_path / "res" / "train_logs" / "sft_val_curve.csv")
    assert [r["step"] for r in curve][:2] == [1, 2] and all(r["n_episodes"] == 9 for r in curve)
    chosen = sel["chosen"]["checkpoint"]
    assert (tmp_path / "sft" / "final" / "adapter_config.json").exists()
    assert (tmp_path / "sft" / "checkpoint-1" / "adapter_model.safetensors").exists()
    assert (tmp_path / "res" / "runs" / "sft_selection.json").exists()
    assert (tmp_path / chosen.split(str(tmp_path) + "/")[-1] / "lexsi_provenance.json").exists()
    stats = json.loads((tmp_path / "res" / "runs" / "sft_dataset_stats.json").read_text())
    assert stats["examples"] > 1000 and stats["dropped_share"] < 0.02


def test_dotted_overrides_reach_nested_config():
    cfg = load_config("configs/sft.yaml", {"mix.injected_oversample": 2.0, "lora.r": 8, "lr": 3e-5})
    assert cfg["mix"] == {"injected_oversample": 2.0, "max_teacher_share": 0.5}
    assert cfg["lora"]["r"] == 8 and cfg["lora"]["alpha"] == 32 and cfg["lr"] == 3e-5
    assert "mix.injected_oversample" not in cfg
