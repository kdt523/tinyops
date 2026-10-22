"""Run a policy over a split; append one Episode per scenario to a jsonl (resumable).

    python -m tinyops.eval.run_eval --policy oracle --split val --out results/episodes/oracle_val.jsonl
    python -m tinyops.eval.run_eval --policy hf --model CohereLabs/tiny-aya-global --split val \
        --batch-size 8 --out results/episodes/tinyaya_zeroshot_val.jsonl
    python -m tinyops.eval.run_eval --policy api --model gemini/gemini-2.5-flash --split test ...

Episodes are appended as soon as they finish, so a Colab disconnect loses at most the episodes in
flight. At the end a manifest (results/runs/<run>.json) and metrics (results/metrics/<run>.*)
are written.
"""

from __future__ import annotations

import csv
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from tinyops.eval.score_episodes import score_file
from tinyops.policies.random_policy import RandomPolicy
from tinyops.policies.scripted import OraclePolicy
from tinyops.scenarios.base import Scenario
from tinyops.scenarios.build_dataset import TEST_SPLITS, load_scenarios, sub_seed
from tinyops.sim.env import EpisodeSession, run_episode
from tinyops.utils.envinfo import collect_env
from tinyops.utils.io import append_jsonl, read_jsonl, write_json
from tinyops.utils.log import get_logger
from tinyops.utils.paths import DATA, RESULTS

log = get_logger(__name__)

PolicyFactory = Callable[[Scenario, int], object]
POLICIES: dict[str, PolicyFactory] = {
    "oracle": lambda sc, seed: OraclePolicy(
        sc.state, random.Random(sub_seed(seed, f"oracle-{sc.id}"))),
    "random": lambda sc, seed: RandomPolicy(random.Random(sub_seed(seed, f"random-{sc.id}"))),
}


def _done_ids(out: Path) -> set[str]:
    return {r["scenario_id"] for r in read_jsonl(out)} if out.exists() else set()


def run_policy(
    scenarios: list[Scenario], factory: PolicyFactory, out: Path, seed: int = 1234,
    max_steps: int = 8,
) -> int:
    """Scripted policies bound to one scenario each. Returns the number of new episodes."""
    done = _done_ids(out)
    n = 0
    for sc in scenarios:
        if sc.id in done:
            continue
        ep = run_episode(sc.state, factory(sc, seed), sc.alert_text, max_steps=max_steps,
                         seed=seed, scenario_id=sc.id)
        append_jsonl(out, ep.model_dump(mode="json"))
        n += 1
    return n


def run_batched(
    scenarios: list[Scenario], policy: Any, out: Path, batch_size: int = 8, seed: int = 1234,
    max_steps: int = 8, meta: dict | None = None,
) -> int:
    """One shared policy. Episodes of a chunk advance in lockstep; finished ones drop out."""
    todo = [sc for sc in scenarios if sc.id not in _done_ids(out)]
    batched = hasattr(policy, "act_batch")
    n = 0
    for i in range(0, len(todo), batch_size):
        active = [
            EpisodeSession(sc.state, sc.alert_text, policy.name, max_steps, seed, sc.id, meta)
            for sc in todo[i : i + batch_size]
        ]
        while active:
            if batched:
                try:
                    outs = policy.act_batch([s.messages for s in active])
                except Exception as e:  # noqa: BLE001 - a failed batch fails its episodes only
                    log.error("batch generation failed: %s", e)
                    for s in active:
                        s.fail(e)
                    outs = [None] * len(active)
            else:
                outs = []
                for s in active:
                    try:
                        outs.append(policy.act(s.messages))
                    except Exception as e:  # noqa: BLE001
                        s.fail(e)
                        outs.append(None)
            for s, o in zip(active, outs, strict=True):
                if o is not None and not s.done:
                    s.apply(o)
            for s in active:
                if s.done:
                    append_jsonl(out, s.episode().model_dump(mode="json"))
                    n += 1
            active = [s for s in active if not s.done]
    return n


def build_policy(
    kind: str, model: str | None = None, adapter: str | None = None, fewshot: int = 0,
    seed: int = 1234, dtype: str = "auto", load_in_4bit: bool = False,
    max_new_tokens: int = 256, rpm: int = 60, name: str | None = None, data: Path = DATA,
):
    if kind == "hf":
        from tinyops.policies.fewshot import build_fewshot_episodes
        from tinyops.policies.hf_policy import HFPolicy

        if not model:
            raise typer.BadParameter("--model is required for --policy hf")
        shots = build_fewshot_episodes(fewshot, seed, data) if fewshot else None
        return HFPolicy(model, adapter, dtype, load_in_4bit, max_new_tokens, shots, name)
    if kind == "api":
        from tinyops.policies.api_policy import APIPolicy

        if not model:
            raise typer.BadParameter("--model is required for --policy api")
        return APIPolicy(model, max_tokens=max_new_tokens, rpm=rpm, name=name)
    raise typer.BadParameter(f"unknown policy {kind!r}; choose oracle, random, hf or api")


def _gpu_name() -> str | None:
    return collect_env()["gpu_name"]


def append_compute_log(results: Path, run: str, gpu: str, hours: float) -> None:
    path = results / "compute_log.csv"
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "run", "gpu", "hours"])
        w.writerow([datetime.now(UTC).date().isoformat(), run, gpu, f"{hours:.4f}"])


def run_eval(
    kind: str, split: str, out: Path, *, model: str | None = None, adapter: str | None = None,
    fewshot: int = 0, seed: int = 1234, limit: int = 0, batch_size: int = 8, max_steps: int = 8,
    dtype: str = "auto", load_in_4bit: bool = False, max_new_tokens: int = 256, rpm: int = 60,
    name: str | None = None, data: Path = DATA, results: Path = RESULTS, score: bool = True,
    final: bool = False,
) -> dict:
    if split in TEST_SPLITS:
        Console(stderr=True).print(
            f"[bold red]WARNING: evaluating on locked split {split!r}. Only base models and the "
            "final model may be evaluated here; never choose anything from these numbers.[/]")
        if adapter and not final:
            raise typer.BadParameter(
                "trained adapters may only be evaluated on test splits with --final (skill 8)")
    scenarios = load_scenarios(split, data)
    if limit:
        scenarios = scenarios[:limit]
    run = out.stem
    start = datetime.now(UTC)
    t0 = time.perf_counter()
    if kind in POLICIES:
        policy_config: dict = {"policy": kind}
        n = run_policy(scenarios, POLICIES[kind], out, seed, max_steps)
    else:
        policy = build_policy(kind, model, adapter, fewshot, seed, dtype, load_in_4bit,
                              max_new_tokens, rpm, name, data)
        policy_config = policy.config()
        meta = {"policy_config": policy_config}
        n = run_batched(scenarios, policy, out, batch_size, seed, max_steps, meta)
        policy_config = policy.config()  # now includes the resolved dtype
    elapsed = time.perf_counter() - t0
    env = collect_env()
    from tinyops.policies.prompt import PROMPT_VERSION

    manifest = {
        "run": run, "split": split, "seed": seed, "max_steps": max_steps,
        "batch_size": batch_size, "limit": limit, "policy_config": policy_config,
        "prompt_version": PROMPT_VERSION, "git_commit": env["git_commit"], "env": env,
        "start_utc": start.isoformat(), "end_utc": datetime.now(UTC).isoformat(),
        "elapsed_s_this_invocation": elapsed, "new_episodes_this_invocation": n,
        "episodes_file": str(out), "scenarios_requested": len(scenarios),
    }
    if env["cuda_available"] and kind in ("hf",):
        manifest["gpu_hours_this_invocation"] = elapsed / 3600
        append_compute_log(results, run, env["gpu_name"] or "gpu", elapsed / 3600)
    write_json(results / "runs" / f"{run}.json", manifest)
    if score:
        score_file(out, split, results / "metrics" / f"{run}.json", data)
    log.info("run %s: %d new episodes in %.1fs", run, n, elapsed)
    return manifest


app = typer.Typer(add_completion=False)


@app.command()
def main(
    policy: Annotated[str, typer.Option(help="oracle | random | hf | api")],
    split: Annotated[str, typer.Option(help="train | val | test | test_injection | test_ood")],
    out: Annotated[Path, typer.Option(help="episodes .jsonl (appended; resumable)")],
    model: Annotated[str | None, typer.Option(help="HF model id or LiteLLM model")] = None,
    adapter: Annotated[str | None, typer.Option(help="LoRA adapter path or hub id")] = None,
    fewshot: Annotated[int, typer.Option(help="k few-shot oracle examples from train")] = 0,
    seed: Annotated[int, typer.Option()] = 1234,
    limit: Annotated[int, typer.Option(help="0 = all scenarios")] = 0,
    batch_size: Annotated[int, typer.Option()] = 8,
    max_steps: Annotated[int, typer.Option()] = 8,
    dtype: Annotated[str, typer.Option(help="auto | float16 | bfloat16 | float32")] = "auto",
    load_in_4bit: Annotated[bool, typer.Option("--load-in-4bit")] = False,
    max_new_tokens: Annotated[int, typer.Option()] = 256,
    rpm: Annotated[int, typer.Option(help="API requests per minute")] = 60,
    name: Annotated[str | None, typer.Option(help="policy name recorded in episodes")] = None,
    data: Annotated[Path, typer.Option()] = DATA,
    results: Annotated[Path, typer.Option()] = RESULTS,
    no_score: Annotated[bool, typer.Option("--no-score")] = False,
    final: Annotated[bool, typer.Option("--final", help="allow adapters on test splits")] = False,
) -> None:
    run_eval(policy, split, out, model=model, adapter=adapter, fewshot=fewshot, seed=seed,
             limit=limit, batch_size=batch_size, max_steps=max_steps, dtype=dtype,
             load_in_4bit=load_in_4bit, max_new_tokens=max_new_tokens, rpm=rpm, name=name,
             data=data, results=results, score=not no_score, final=final)


if __name__ == "__main__":
    app()
