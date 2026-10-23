"""Stage B: GRPO with TinyOps rewards through AgentTune's `create_agentic_trainer`.

    python -m tinyops.train.grpo --config configs/grpo.yaml [--set max_steps=50]

How the pieces fit (verified in the source, docs/api_notes.md and DECISIONS.md):
- we build the rollout function ourselves (`create_rollout_fn`) and pass it as `rollout_func`, so we
  control `pre_step_hook` (binds the scenario for the tools) and the system prompt;
- rewards replay each rollout's assistant turns through the strict KubeSim environment;
- the SFT adapter is merged into the base first; GRPO trains a fresh LoRA on top, so the KL
  reference (adapter disabled) is the SFT policy.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Annotated

import typer
import yaml

from tinyops.policies.prompt import RULES_PROMPT
from tinyops.scenarios.build_dataset import load_scenarios
from tinyops.scoring.rewards import RewardSet
from tinyops.train.merge import merge_adapter
from tinyops.train.rl_data import build_rl_scenarios, load_type_weights, rl_dataset
from tinyops.train.rollout_env import ScenarioRegistry, make_pre_step_hook, make_rollout_tools
from tinyops.train.sft import load_config, write_run_provenance
from tinyops.train.stage_b import select_vs_reference
from tinyops.train.val_eval import make_val_callback, read_curve, select_val_subset
from tinyops.utils.envinfo import collect_env
from tinyops.utils.io import read_json, read_jsonl, write_json
from tinyops.utils.log import get_logger
from tinyops.utils.seed import set_seed

log = get_logger(__name__)


def peek_rewards(log_path: Path, k: int = 10, seed: int = 0) -> list[str]:
    """k random recent rollouts with their reward breakdown, as readable lines (read them!)."""
    if not log_path.exists():
        return []
    rows = read_jsonl(log_path)[-200:]
    picks = random.Random(seed).sample(rows, min(k, len(rows)))
    out = []
    for r in picks:
        comps = " ".join(f"{n}={r[n]:+.2f}" for n in ("diagnosis", "format", "schema", "efficiency", "grounding", "safety"))
        turns = " | ".join(t[:90].replace("\n", " ") for t in r["texts"][:5])
        out.append(f"[{r['scenario_id']}] total={r['total']:+.2f} {comps}\n    {turns}")
    return out


def reward_peek_callback(log_path: Path, every: int):
    from transformers import TrainerCallback

    class Peek(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if every and state.global_step % every == 0:
                for line in peek_rewards(log_path, 10, state.global_step):
                    log.info("%s", line)

    return Peek()


def sft_reference(path: str | None) -> dict | None:
    if path and Path(path).exists():
        return read_json(path)["chosen"]
    return None


def run_grpo(cfg: dict) -> dict:
    import torch
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn
    from agenttune.agentic.trajectory.dataset import trajectory_writer
    from agenttune.core.backend_factory import create_agentic_trainer
    from transformers import AutoTokenizer

    set_seed(cfg["seed"])
    data, results, out_dir = Path(cfg["data_dir"]), Path(cfg["results_dir"]), Path(cfg["output_dir"])
    model_path = (str(merge_adapter(cfg["base_model"], cfg["start_from"], cfg["merged_dir"]))
                  if cfg.get("start_from") else cfg["base_model"])

    weights = load_type_weights(cfg.get("sft_val_metrics"))
    scenarios = build_rl_scenarios(cfg["n_prompts"], cfg["rl_seed"], weights, data)
    by_id = {s.id: s for s in scenarios}
    train_ds = rl_dataset(scenarios)
    log.info("RL prompts: %d (type weights %s)", len(scenarios), {k: round(v, 2) for k, v in weights.items()})

    reward_log = results / "train_logs" / "grpo_rewards.jsonl"
    reward_log.unlink(missing_ok=True)
    rewards = RewardSet(by_id, cfg["reward_weights"], log_path=reward_log)
    tools = make_rollout_tools()
    bound: list[str] = []
    rollout_fn = create_rollout_fn(
        tools=tools, max_steps=cfg["max_steps_per_turn"], system_prompt=RULES_PROMPT,
        pre_step_hook=make_pre_step_hook(ScenarioRegistry(scenarios), bound),
        on_trajectory_end=trajectory_writer(out_dir / "trajectories.jsonl"),
    )

    tok = AutoTokenizer.from_pretrained(model_path)
    subset = select_val_subset(load_scenarios("val", data), per_type=cfg["val_subset"] // 9, seed=cfg["seed"])
    curve = results / "train_logs" / "grpo_val_curve.csv"
    curve.unlink(missing_ok=True)
    callbacks = [make_val_callback(cfg, subset, tok, curve, "grpo"),
                 reward_peek_callback(reward_log, cfg["peek_every_steps"])]

    cuda = torch.cuda.is_available()
    dtype = (torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else
             torch.float16 if cuda else torch.float32) if cfg["dtype"] == "auto" else getattr(torch, cfg["dtype"])
    lora = cfg["lora"]
    trainer = create_agentic_trainer(
        "grpo", backend="trl", model=model_path, reward_funcs=rewards.reward_funcs(),
        train_dataset=train_ds, tools=tools, rollout_func=rollout_fn,
        peft_config={"r": lora["r"], "lora_alpha": lora["alpha"],
                     "target_modules": lora["target_modules"], "task_type": "CAUSAL_LM"},
        output_dir=str(out_dir), max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["batch"], gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"], num_generations=cfg["num_generations"],
        max_completion_length=cfg["max_completion_length"], temperature=cfg["temperature"],
        max_steps_per_turn=cfg["max_steps_per_turn"], use_vllm=False, log_completions=True,
        save_steps=cfg["eval_every_steps"], report_to="none", seed=cfg["seed"],
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16 and cuda,
        model_init_kwargs={"dtype": dtype}, trajectories_file="trajectories.jsonl",
        push_to_hub=bool(cfg.get("push_to_hub")), hub_model_id=cfg.get("push_to_hub"),
        callbacks=callbacks,
    )

    t0 = time.perf_counter()
    trainer.train()
    hours = (time.perf_counter() - t0) / 3600

    ref = sft_reference(cfg.get("sft_selection"))
    rows = read_curve(curve)
    if ref is None:
        log.warning("no SFT selection file: choosing by val success only")
        ref = {"unsafe_rate": 1.0, "format_valid_rate": 0.0}
    sel = select_vs_reference(rows, ref)
    write_json(results / "runs" / "grpo_selection.json", sel)
    if sel["chosen"]:
        write_run_provenance(Path(sel["chosen"]["checkpoint"]), {**cfg, "lora": {**lora, "dropout": 0.0},
                             "epochs": None, "max_seq_len": None, "mode": "grpo"}, {"rl_prompts": len(scenarios)})
    write_json(results / "runs" / "grpo_train_run.json", {
        "config": cfg, "env": collect_env(), "train_hours": hours, "beta": getattr(trainer.trainer.args, "beta", None),
        "rollouts_bound": len(bound), "unbound_rollouts": bound.count("<unbound>"),
        "selection": sel["chosen"], "keep_sft": sel["keep_reference"]})
    if cuda:
        from tinyops.eval.run_eval import append_compute_log

        append_compute_log(results, "grpo_train", collect_env()["gpu_name"] or "gpu", hours)
    return sel


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Annotated[Path, typer.Option()] = Path("configs/grpo.yaml"),
    set_: Annotated[list[str] | None, typer.Option("--set", help="override, e.g. --set max_steps=50")] = None,
) -> None:
    overrides = {}
    for item in set_ or []:
        k, _, v = item.partition("=")
        overrides[k] = yaml.safe_load(v)
    run_grpo(load_config(config, overrides))


if __name__ == "__main__":
    app()
