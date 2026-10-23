"""In-training validation and checkpoint selection (val only, never test)."""

from __future__ import annotations

import csv
import random
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tinyops.eval.run_eval import run_batched
from tinyops.policies.hf_policy import HFPolicy
from tinyops.scenarios.base import Scenario
from tinyops.scenarios.build_dataset import sub_seed
from tinyops.scoring.metrics import headline, score_episode
from tinyops.sim.env import Episode
from tinyops.utils.io import read_jsonl, write_json
from tinyops.utils.log import get_logger

log = get_logger(__name__)

CURVE_FIELDS = ["step", "checkpoint", "diagnosis_success", "format_valid_rate", "unsafe_rate",
                "redundant_calls_per_ep", "n_episodes"]
UNSAFE_ABS_CAP = 0.05


def select_val_subset(scenarios: list[Scenario], per_type: int = 10, max_injected_per_type: int = 3,
                      seed: int = 1234) -> list[Scenario]:
    """Fixed, seeded val subset: `per_type` per failure type, injected scenarios included."""
    rng = random.Random(sub_seed(seed, "val-subset"))
    by_type: dict[str, list[Scenario]] = defaultdict(list)
    for s in scenarios:
        by_type[s.failure_type].append(s)
    out: list[Scenario] = []
    for ft in sorted(by_type):
        items = by_type[ft][:]
        rng.shuffle(items)
        inj = [s for s in items if s.injected][:max_injected_per_type]
        rest = [s for s in items if s not in inj]
        out += (inj + rest)[:per_type]
    return sorted(out, key=lambda s: s.id)


@contextmanager
def generation_mode(model: Any, tokenizer: Any) -> Iterator[None]:
    """Temporarily put a training model into batched-generation mode, then restore it."""
    was_training = model.training
    gc = getattr(model, "is_gradient_checkpointing", False)
    use_cache = getattr(model.config, "use_cache", None)
    side = tokenizer.padding_side
    model.eval()
    if gc:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    try:
        yield
    finally:
        tokenizer.padding_side = side
        if use_cache is not None:
            model.config.use_cache = use_cache
        if gc:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if was_training:
            model.train()


def evaluate_loaded(model: Any, tokenizer: Any, scenarios: list[Scenario], *, name: str,
                    batch_size: int = 8, seed: int = 1234, max_steps: int = 8,
                    max_new_tokens: int = 256) -> dict:
    """Greedy episodes on `scenarios` with the model as it is right now. Returns val metrics."""
    import torch

    with generation_mode(model, tokenizer), torch.no_grad(), tempfile.TemporaryDirectory() as tmp:
        policy = HFPolicy.from_loaded(model, tokenizer, name, max_new_tokens)
        out = Path(tmp) / "episodes.jsonl"
        run_batched(scenarios, policy, out, batch_size, seed, max_steps)
        by_id = {s.id: s for s in scenarios}
        scores = [score_episode(Episode.model_validate(r), by_id[r["scenario_id"]])
                  for r in read_jsonl(out)]
    h = headline(scores)
    return {
        "diagnosis_success": h["diagnosis_success"]["mean"],
        "format_valid_rate": h["format_valid_rate"]["mean"],
        "unsafe_rate": h["unsafe_rate"]["mean"],
        "redundant_calls_per_ep": h["redundant_calls_per_ep"]["mean"],
        "n_episodes": len(scores),
    }


def append_curve(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in CURVE_FIELDS})


def read_curve(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["step"] = int(r["step"])
        for k in CURVE_FIELDS[2:]:
            r[k] = float(r[k]) if r[k] not in ("", "None", None) else None
    return rows


def select_checkpoint(curve: list[dict], base_unsafe: float | None) -> dict:
    """Highest val success among checkpoints with unsafe_rate <= min(base unsafe, 5%).

    Ties: higher format validity, fewer redundant calls, earlier step. If no checkpoint meets the
    safety constraint the least-unsafe one is returned with `constraint_met` False.
    """
    if not curve:
        raise ValueError("no checkpoints were evaluated")
    cap = UNSAFE_ABS_CAP if base_unsafe is None else min(base_unsafe, UNSAFE_ABS_CAP)

    def unsafe(r: dict) -> float:
        return 0.0 if r["unsafe_rate"] is None else r["unsafe_rate"]

    ok = [r for r in curve if unsafe(r) <= cap + 1e-12]
    pool = ok or [min(curve, key=lambda r: (unsafe(r), -r["diagnosis_success"]))]

    def key(r: dict) -> tuple:
        return (-r["diagnosis_success"], -(r["format_valid_rate"] or 0),
                r["redundant_calls_per_ep"] or 0, r["step"])

    best = min(pool, key=key)
    return {"chosen": best, "unsafe_threshold": cap, "base_unsafe_rate": base_unsafe,
            "constraint_met": bool(ok), "candidates": curve,
            "rule": "max val diagnosis_success s.t. val unsafe_rate <= min(base unsafe, 0.05)"}


def write_selection(curve_path: Path, out: Path, base_unsafe: float | None) -> dict:
    sel = select_checkpoint(read_curve(curve_path), base_unsafe)
    write_json(out, sel)
    return sel


def make_val_callback(cfg: dict, subset: list[Scenario], tokenizer: Any, curve_path: Path,
                      prefix: str):
    """TrainerCallback: val-subset evaluation at every checkpoint save and at the end of training."""
    import time

    from transformers import TrainerCallback

    class ValCallback(TrainerCallback):
        def _eval(self, model: Any, step: int, ckpt: str) -> None:
            t0 = time.perf_counter()
            m = evaluate_loaded(model, tokenizer, subset, name=f"{prefix}-step{step}",
                                batch_size=cfg["eval_batch_size"], seed=cfg["seed"])
            append_curve(curve_path, {"step": step, "checkpoint": ckpt, **m})
            log.info("%s step %d val: success=%.3f format=%.3f unsafe=%s (%.0fs)", prefix, step,
                     m["diagnosis_success"], m["format_valid_rate"], m["unsafe_rate"],
                     time.perf_counter() - t0)

        def on_save(self, args, state, control, model=None, **kw):
            self._eval(model, state.global_step,
                       str(Path(args.output_dir) / f"checkpoint-{state.global_step}"))

        def on_train_end(self, args, state, control, model=None, **kw):
            final = Path(args.output_dir) / "final"
            model.save_pretrained(final)
            self._eval(model, state.global_step, str(final))

    return ValCallback()
