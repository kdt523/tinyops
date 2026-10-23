"""Stage A: LoRA SFT on gold trajectories with assistant-only loss and val-based selection.

    python -m tinyops.train.sft --config configs/sft.yaml

Uses TRL's SFTTrainer + peft on our own pre-tokenized, assistant-masked dataset. AgentTune's
`create_distill_trainer` is not used: it renders tool calls as `<tool_call>{...}</tool_call>`
(not Tiny Aya's fallback list format) and needs full-tier EventLogs (docs/api_notes.md section 5).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from tinyops.scenarios.build_dataset import load_scenarios, run_gold
from tinyops.sim.env import Episode
from tinyops.train.sft_data import MixConfig, build_sft_dataset
from tinyops.train.val_eval import (
    CURVE_FIELDS,
    make_val_callback,
    select_val_subset,
    write_selection,
)
from tinyops.utils.envinfo import collect_env
from tinyops.utils.io import read_json, read_jsonl, write_json
from tinyops.utils.log import get_logger
from tinyops.utils.seed import set_seed

log = get_logger(__name__)


def load_config(path: Path | str, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = key.split(".")  # "mix.injected_oversample" reaches into the nested dict
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return cfg


def ensure_gold(data: Path, seed: int) -> None:
    for split in ("train", "val"):
        if not (data / split / "gold_episodes.jsonl").exists():
            log.info("gold episodes for %s missing: generating (deterministic)", split)
            run_gold(split, seed, data)


def load_gold(data: Path, split: str) -> tuple[list[Episode], dict]:
    eps = [Episode.model_validate(r) for r in read_jsonl(data / split / "gold_episodes.jsonl")]
    scenarios = {s.id: s for s in load_scenarios(split, data)}
    return eps, scenarios


def base_unsafe_rate(path: str | None) -> float | None:
    if path and Path(path).exists():
        return read_json(path)["headline"]["unsafe_rate"]["mean"]
    log.warning("no base metrics at %s: selection uses only the 5%% absolute unsafe cap", path)
    return None


def write_run_provenance(out_dir: Path, cfg: dict, stats: dict) -> dict:
    params = {"lora": cfg["lora"], "lr": cfg["lr"], "epochs": cfg["epochs"],
              "max_seq_len": cfg["max_seq_len"], "seed": cfg["seed"], "mode": cfg["mode"],
              "dataset_stats": stats}
    try:
        from agenttune.utils.provenance import write_provenance

        return write_provenance(out_dir, "sft_lora", cfg["base_model"], str(Path(cfg["data_dir"]) / "train"),
                                params=params)
    except ImportError:  # AgentTune is a [train] extra; same schema, written by hand
        record = {"schema": "lexsi.provenance/1", "library": "tinyops (agenttune unavailable)",
                  "base_model": cfg["base_model"], "method": "sft_lora", "params": params,
                  "inputs": [{"kind": "dataset", "ref": str(Path(cfg["data_dir"]) / "train")}]}
        write_json(out_dir / "lexsi_provenance.json", record)
        return record


def warmup_kwargs(config_cls: Any, ratio: float) -> dict:
    """transformers >= 5 takes a float `warmup_steps` in [0,1) as a ratio; older ones `warmup_ratio`."""
    import inspect

    params = inspect.signature(config_cls.__init__).parameters
    return {"warmup_ratio": ratio} if "warmup_ratio" in params else {"warmup_steps": ratio}


def run_sft(cfg: dict) -> dict:
    """Train, evaluate on the val subset along the way, select a checkpoint. Returns the selection."""
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    set_seed(cfg["seed"])
    data, results = Path(cfg["data_dir"]), Path(cfg["results_dir"])
    out_dir = Path(cfg["output_dir"])
    ensure_gold(data, cfg["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    gold, train_scs = load_gold(data, "train")
    extra_rows = []
    if cfg.get("teacher_episodes"):
        from tinyops.train.teacher_data import teacher_rows

        teachers = [Episode.model_validate(r) for r in read_jsonl(cfg["teacher_episodes"])]
        extra = {s.id: s for s in load_scenarios("train_extra", data)}
        extra_rows = teacher_rows(teachers, extra)
    mix = MixConfig(**cfg.get("mix", {}))
    train_ds, stats = build_sft_dataset(gold, train_scs, cfg["mode"], tokenizer=tok,
                                        max_seq_len=cfg["max_seq_len"], mix=mix,
                                        extra_rows=extra_rows, seed=cfg["seed"])
    log.info("SFT dataset: %s", json.dumps({k: v for k, v in stats.items() if k != "by_failure_type"}))
    if stats["dropped_share"] > 0.02:
        log.warning("%.1f%% of examples exceed max_seq_len=%d (plan expected < 2%%)",
                    100 * stats["dropped_share"], cfg["max_seq_len"])
    train_ds = train_ds.remove_columns("id")
    write_json(results / "runs" / "sft_dataset_stats.json", stats)

    cuda = torch.cuda.is_available()
    if cfg["dtype"] == "auto":
        bf16 = cuda and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    else:
        dtype = getattr(torch, cfg["dtype"])
        bf16 = dtype == torch.bfloat16
    fp16 = dtype == torch.float16 and cuda

    val_scs = load_scenarios("val", data)
    subset = select_val_subset(val_scs, per_type=cfg["val_subset"] // 9, seed=cfg["seed"])
    curve = results / "train_logs" / "sft_val_curve.csv"
    curve.unlink(missing_ok=True)

    lora = cfg["lora"]
    args = SFTConfig(
        output_dir=str(out_dir), num_train_epochs=cfg["epochs"], max_steps=cfg.get("max_steps", -1),
        learning_rate=cfg["lr"], lr_scheduler_type=cfg["scheduler"], **warmup_kwargs(SFTConfig, cfg["warmup_ratio"]),
        per_device_train_batch_size=cfg["per_device_batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16, fp16=fp16, logging_steps=10, save_strategy="steps",
        save_steps=cfg["eval_every_steps"], report_to="none", seed=cfg["seed"],
        max_length=cfg["max_seq_len"], packing=False, remove_unused_columns=False,
        model_init_kwargs={"dtype": dtype},
        push_to_hub=bool(cfg.get("push_to_hub")), hub_model_id=cfg.get("push_to_hub"),
        hub_strategy="every_save",
    )
    trainer = SFTTrainer(
        model=cfg["base_model"], args=args, train_dataset=train_ds, processing_class=tok,
        peft_config=LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                               target_modules=lora["target_modules"], task_type="CAUSAL_LM"),
        callbacks=[make_val_callback(cfg, subset, tok, curve, "sft")],
    )
    t0 = time.perf_counter()
    trainer.train()
    hours = (time.perf_counter() - t0) / 3600
    sel = write_selection(curve, results / "runs" / "sft_selection.json",
                          base_unsafe_rate(cfg.get("base_metrics")))
    write_run_provenance(Path(sel["chosen"]["checkpoint"]), cfg, stats)
    write_json(results / "runs" / "sft_train_run.json", {
        "config": cfg, "env": collect_env(), "train_hours": hours, "dataset_stats": stats,
        "curve_fields": CURVE_FIELDS, "selection": sel["chosen"]})
    if cuda:
        from tinyops.eval.run_eval import append_compute_log

        append_compute_log(results, "sft_train", collect_env()["gpu_name"] or "gpu", hours)
    log.info("selected %s (constraint met: %s)", sel["chosen"]["checkpoint"], sel["constraint_met"])
    return sel


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="configs/sft.yaml")] = Path("configs/sft.yaml"),
    set_: Annotated[list[str] | None, typer.Option("--set", help="override, e.g. --set lr=5e-5")] = None,
) -> None:
    overrides = {}
    for item in set_ or []:
        k, _, v = item.partition("=")
        overrides[k] = yaml.safe_load(v)
    run_sft(load_config(config, overrides))


if __name__ == "__main__":
    app()
