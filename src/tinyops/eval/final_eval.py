"""The one official evaluation on the locked test splits.

    python -m tinyops.eval.final_eval --config configs/final.yaml [--only final --only sft]

1. verifies data/TEST_HASHES.txt and aborts on a mismatch;
2. runs every policy on test / test_injection / test_ood (greedy, seed 1234, max_steps 8), or reuses
   skill-5 baseline episodes when they were made with the same prompt version, seed and step limit;
3. writes results/final/episodes/<policy>_<split>.jsonl and results/final/metrics/<policy>_<split>.json;
4. writes results/final/FINAL_RUN.json (git commit, adapter revisions, hashes). A second run is
   refused unless you pass --rerun-reason; justify it in docs/DECISIONS.md and re-run ALL policies.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml

from tinyops.eval.run_eval import run_eval
from tinyops.eval.score_episodes import load_split_scenarios, score_file
from tinyops.policies.hf_policy import split_adapters
from tinyops.policies.prompt import PROMPT_VERSION
from tinyops.scenarios.build_dataset import verify_test_hashes
from tinyops.utils.envinfo import collect_env
from tinyops.utils.io import read_json, read_jsonl, sha256_file, write_json
from tinyops.utils.log import get_logger
from tinyops.utils.paths import DATA, RESULTS

log = get_logger(__name__)
FINAL = RESULTS / "final"


def check_config(cfg: dict) -> None:
    for name, spec in cfg["policies"].items():
        if "<user>" in str(spec.get("adapter", "")) or "<user>" in str(spec.get("model", "")):
            raise ValueError(f"policy {name!r} still has a <user> placeholder: {spec}")
        if spec["kind"] not in ("hf", "api"):
            raise ValueError(f"policy {name!r}: kind must be hf or api")


def can_reuse(src_episodes: Path, src_manifest: Path, split: str, cfg: dict, data: Path) -> str | None:
    """None if the baseline episodes may be reused, else the reason they may not."""
    if not src_episodes.exists() or not src_manifest.exists():
        return "baseline files not found"
    man = read_json(src_manifest)
    if man.get("prompt_version") != PROMPT_VERSION:
        return f"prompt version {man.get('prompt_version')!r} != {PROMPT_VERSION!r}"
    if man.get("seed") != cfg["seed"] or man.get("max_steps") != cfg["max_steps"]:
        return "seed or max_steps differ"
    if man.get("split") != split or man.get("limit"):
        return "baseline was a different split or a partial run"
    ids = {r["scenario_id"] for r in read_jsonl(src_episodes)}
    if ids != set(load_split_scenarios(split, data)):
        return "baseline does not cover the whole split"
    return None


def hub_revision(adapter_spec: str) -> dict[str, str | None]:
    """Hub commit sha of each adapter (None for local paths or when offline)."""
    out: dict[str, str | None] = {}
    for a in split_adapters(adapter_spec):
        if Path(a).exists():
            out[a] = None
            continue
        try:
            from huggingface_hub import HfApi

            out[a] = HfApi().model_info(a).sha
        except Exception:  # noqa: BLE001 - offline / private / not a hub id
            out[a] = None
    return out


def run_final(
    cfg: dict, only: list[str] | None = None, data: Path = DATA, results: Path = FINAL,
    baselines: Path = RESULTS, rerun_reason: str | None = None, resolve_revisions: bool = True,
) -> dict:
    verify_test_hashes(data)  # aborts (raises) on any mismatch
    check_config(cfg)
    lock = results / "FINAL_RUN.json"
    if lock.exists() and not rerun_reason:
        raise FileExistsError(f"{lock} exists: the final run is locked. A re-run needs --rerun-reason "
                              "(log it in docs/DECISIONS.md and re-run all policies).")
    done: dict[str, dict] = {}
    for name, spec in cfg["policies"].items():
        if only and name not in only:
            continue
        for split in spec.get("splits", cfg["splits"]):
            out = results / "episodes" / f"{name}_{split}.jsonl"
            reuse = spec.get("reuse_from")
            if reuse and not out.exists():
                src = reuse.format(split=split)
                why = can_reuse(baselines / "episodes" / f"{src}.jsonl", baselines / "runs" / f"{src}.json",
                                split, cfg, data)
                if why is None:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(baselines / "episodes" / f"{src}.jsonl", out)
                    score_file(out, split, results / "metrics" / f"{name}_{split}.json", data)
                    done[f"{name}_{split}"] = {"reused_from": src}
                    log.info("reused %s for %s_%s", src, name, split)
                    continue
                log.warning("not reusing %s for %s_%s: %s", src, name, split, why)
            m = run_eval(
                spec["kind"], split, out, model=spec.get("model", cfg["model"]),
                adapter=spec.get("adapter"), fewshot=spec.get("fewshot", 0), seed=cfg["seed"],
                batch_size=cfg["batch_size"], max_steps=cfg["max_steps"],
                max_new_tokens=spec.get("max_new_tokens", cfg["max_new_tokens"]),
                name=name, data=data, results=results, final=True)
            done[f"{name}_{split}"] = {"manifest": f"runs/{name}_{split}.json",
                                       "gpu_hours": m.get("gpu_hours_this_invocation")}
    revisions = {name: hub_revision(spec["adapter"]) if resolve_revisions else {}
                 for name, spec in cfg["policies"].items() if spec.get("adapter")}
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(), "git_commit": collect_env()["git_commit"],
        "prompt_version": PROMPT_VERSION, "seed": cfg["seed"], "max_steps": cfg["max_steps"],
        "test_hashes": (data / "TEST_HASHES.txt").read_text(encoding="utf-8"),
        "policies": cfg["policies"], "adapter_revisions": revisions, "runs": done,
        "episodes_sha256": {str(p.relative_to(results)): sha256_file(p)
                            for p in sorted((results / "episodes").glob("*.jsonl"))},
        "rerun_reason": rerun_reason,
    }
    expected = [f"{n}_{sp}" for n, spec in cfg["policies"].items()
                for sp in spec.get("splits", cfg["splits"])]
    missing = [e for e in expected if not (results / "episodes" / f"{e}.jsonl").exists()]
    record["missing"] = missing
    # the lock is written only when EVERY policy x split exists; partial sessions stay re-runnable
    write_json(results / ("FINAL_RUN.partial.json" if missing else "FINAL_RUN.json"), record)
    if missing:
        log.warning("final run incomplete, still missing: %s", missing)
    return record


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Annotated[Path, typer.Option()] = Path("configs/final.yaml"),
    only: Annotated[list[str] | None, typer.Option(help="run only these policies (GPU session split)")] = None,
    rerun_reason: Annotated[str | None, typer.Option()] = None,
    data: Annotated[Path, typer.Option()] = DATA,
    results: Annotated[Path, typer.Option()] = FINAL,
) -> None:
    run_final(yaml.safe_load(config.read_text(encoding="utf-8")), only, data, results,
              rerun_reason=rerun_reason)


if __name__ == "__main__":
    app()
