"""Stage B fallback: offline DPO with turn-level preference pairs.

chosen = the gold (oracle) assistant turn, rejected = what a failing policy said at the first turn
where it differs from gold, both after the identical conversation prefix. Failures that matter
most come first: obeyed injections, loops, malformed/unparseable turns, then other wrong answers.
Only TRAIN scenarios are used.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Annotated

import typer
import yaml

from tinyops.policies.prompt import build_system_prompt, format_tool_result
from tinyops.scenarios.base import Scenario
from tinyops.scenarios.build_dataset import load_scenarios, sub_seed
from tinyops.scoring.metrics import score_episode
from tinyops.sim.env import Episode
from tinyops.sim.export import step_text
from tinyops.train.merge import merge_adapter
from tinyops.train.sft import load_config, write_run_provenance
from tinyops.train.stage_b import select_vs_reference
from tinyops.train.val_eval import make_val_callback, read_curve, select_val_subset
from tinyops.utils.envinfo import collect_env
from tinyops.utils.io import read_json, read_jsonl, write_json
from tinyops.utils.log import get_logger
from tinyops.utils.seed import set_seed

log = get_logger(__name__)
PRIORITY = {"obeyed_injection": 0, "loop": 1, "malformed": 2, "wrong_answer": 3}


def failure_kind(ep: Episode, sc: Scenario) -> str | None:
    """None if the episode succeeded safely; else the most important way it failed."""
    s = score_episode(ep, sc)
    if s.unsafe and s.injected and s.injection_family != "benign_lookalike":
        return "obeyed_injection"
    if s.success and not s.unsafe:
        return None
    if s.looped:
        return "loop"
    if s.parse_failure_stop or any(not st.parsed_ok for st in ep.steps):
        return "malformed"
    return "wrong_answer"


def first_divergence_pair(policy: Episode, gold: Episode, sc: Scenario) -> dict | None:
    """Pair at the first policy turn that differs from gold, after a shared prefix."""
    prompt = [{"role": "system", "content": build_system_prompt()},
              {"role": "user", "content": sc.alert_text}]
    for g, p in zip(gold.steps, policy.steps, strict=False):
        gt, pt = step_text(g), step_text(p)
        if gt != pt:
            return {"prompt": prompt, "chosen": [{"role": "assistant", "content": gt}],
                    "rejected": [{"role": "assistant", "content": pt}], "scenario_id": sc.id}
        prompt = [*prompt, {"role": "assistant", "content": gt},
                  {"role": "user", "content": format_tool_result(g.tool_name or "parser", g.observation)}]
    return None  # identical so far: the policy only stopped early or ran longer, no clean pair


def build_dpo_pairs(policy_episodes: list[Episode], gold: dict[str, Episode],
                    scenarios: dict[str, Scenario], max_pairs: int = 1500, seed: int = 1234) -> list[dict]:
    rng = random.Random(sub_seed(seed, "dpo-pairs"))
    ranked = []
    for ep in policy_episodes:
        sc = scenarios[ep.scenario_id]
        if sc.split != "train" or ep.scenario_id not in gold:
            continue  # train data only
        kind = failure_kind(ep, sc)
        if kind is None:
            continue
        pair = first_divergence_pair(ep, gold[ep.scenario_id], sc)
        if pair is not None:
            ranked.append((PRIORITY[kind], rng.random(), {**pair, "kind": kind}))
    ranked.sort(key=lambda r: (r[0], r[1]))
    return [r[2] for r in ranked[:max_pairs]]


def run_dpo(cfg: dict) -> dict:
    import torch
    from agenttune.core.backend_factory import create_agentic_trainer
    from datasets import Dataset
    from transformers import AutoTokenizer

    set_seed(cfg["seed"])
    data, results, out_dir = Path(cfg["data_dir"]), Path(cfg["results_dir"]), Path(cfg["output_dir"])
    model_path = (str(merge_adapter(cfg["base_model"], cfg["start_from"], cfg["merged_dir"]))
                  if cfg.get("start_from") else cfg["base_model"])
    scenarios = {s.id: s for s in load_scenarios("train", data)}
    gold = {e["scenario_id"]: Episode.model_validate(e)
            for e in read_jsonl(data / "train" / "gold_episodes.jsonl")}
    policy = [Episode.model_validate(r) for f in cfg["policy_episodes"] for r in read_jsonl(f)]
    pairs = build_dpo_pairs(policy, gold, scenarios, cfg["max_pairs"], cfg["seed"])
    if not pairs:
        raise ValueError("no preference pairs: run base/SFT on the train split first (see the notebook)")
    kinds = {k: sum(p["kind"] == k for p in pairs) for k in PRIORITY}
    log.info("DPO pairs: %d %s", len(pairs), kinds)
    write_json(results / "runs" / "dpo_pairs_stats.json", {"pairs": len(pairs), "by_kind": kinds})
    ds = Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs])

    cuda = torch.cuda.is_available()
    dtype = (torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else
             torch.float16 if cuda else torch.float32) if cfg["dtype"] == "auto" else getattr(torch, cfg["dtype"])
    lora = cfg["lora"]
    tok = AutoTokenizer.from_pretrained(model_path)
    subset = select_val_subset(load_scenarios("val", data), per_type=cfg["val_subset"] // 9, seed=cfg["seed"])
    curve = results / "train_logs" / "dpo_val_curve.csv"
    curve.unlink(missing_ok=True)
    callbacks = [make_val_callback(cfg, subset, tok, curve, "dpo")]
    trainer = create_agentic_trainer(
        "dpo", backend="trl", model=model_path, train_dataset=ds,
        peft_config={"r": lora["r"], "lora_alpha": lora["alpha"],
                     "target_modules": lora["target_modules"], "task_type": "CAUSAL_LM"},
        output_dir=str(out_dir), beta=cfg["beta"], learning_rate=cfg["lr"],
        num_train_epochs=cfg["epochs"], per_device_train_batch_size=cfg["batch"],
        gradient_accumulation_steps=cfg["grad_accum"], max_length=cfg["max_length"],
        save_steps=cfg["eval_every_steps"], report_to="none", seed=cfg["seed"],
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16 and cuda,
        model_init_kwargs={"dtype": dtype}, use_rollouts=False, callbacks=callbacks,
    )
    t0 = time.perf_counter()
    trainer.train()
    hours = (time.perf_counter() - t0) / 3600
    ref = read_json(cfg["sft_selection"])["chosen"] if Path(cfg["sft_selection"]).exists() else \
        {"unsafe_rate": 1.0, "format_valid_rate": 0.0}
    sel = select_vs_reference(read_curve(curve), ref)
    write_json(results / "runs" / "dpo_selection.json", sel)
    if sel["chosen"]:
        write_run_provenance(Path(sel["chosen"]["checkpoint"]), {**cfg, "lora": {**lora, "dropout": 0.0},
                             "epochs": cfg["epochs"], "max_seq_len": cfg["max_length"], "mode": "dpo"},
                             {"pairs": len(pairs), "by_kind": kinds})
    write_json(results / "runs" / "dpo_train_run.json", {
        "config": cfg, "env": collect_env(), "train_hours": hours, "selection": sel["chosen"],
        "keep_sft": sel["keep_reference"]})
    if cuda:
        from tinyops.eval.run_eval import append_compute_log

        append_compute_log(results, "dpo_train", collect_env()["gpu_name"] or "gpu", hours)
    return sel


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Annotated[Path, typer.Option()] = Path("configs/dpo.yaml"),
    set_: Annotated[list[str] | None, typer.Option("--set")] = None,
) -> None:
    overrides = {}
    for item in set_ or []:
        k, _, v = item.partition("=")
        overrides[k] = yaml.safe_load(v)
    run_dpo(load_config(config, overrides))


if __name__ == "__main__":
    app()
