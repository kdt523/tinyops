"""Push cards, adapters, data and the Space to the Hugging Face Hub (idempotent, dry-run by default).

    python -m tinyops.report.publish plan --user YOUR_HF_USERNAME
    python -m tinyops.report.publish push --user YOUR_HF_USERNAME --execute

Publishing is outward-facing, so nothing is uploaded without `--execute`. Every file is scanned for
tokens and keys first; `.env` and optimizer state are never uploaded.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer

from tinyops.scenarios.build_dataset import SPLITS
from tinyops.utils.io import read_json
from tinyops.utils.log import get_logger
from tinyops.utils.paths import DATA, RESULTS, ROOT

log = get_logger(__name__)

SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"), re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?:HF_TOKEN|GEMINI_API_KEY)\s*=\s*[\"']?[A-Za-z0-9_\-]{16,}"),
]
NEVER_UPLOAD = {".env", "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"}
SUBSTITUTE_SUFFIXES = {".md", ".py", ".txt"}
TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".csv", ".txt", ".py", ".yaml", ".yml", ".toml", ".cfg"}


@dataclass
class Item:
    repo_id: str
    repo_type: str  # model | dataset | space
    local: Path
    remote: str
    subst: bool = False  # replace the <user> placeholder in this text file when uploading


@dataclass
class Plan:
    user: str = ""
    items: list[Item] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def repos(self) -> dict[tuple[str, str], list[Item]]:
        out: dict[tuple[str, str], list[Item]] = {}
        for it in self.items:
            out.setdefault((it.repo_id, it.repo_type), []).append(it)
        return out


def _add(plan: Plan, repo: str, rtype: str, local: Path, remote: str, subst: bool = False) -> None:
    if local.name in NEVER_UPLOAD:
        plan.skipped.append(f"{local} (never uploaded)")
    elif local.exists():
        plan.items.append(Item(repo, rtype, local, remote, subst))
    else:
        plan.skipped.append(f"{local} (missing)")


def _add_tree(plan: Plan, repo: str, rtype: str, root: Path, prefix: str, patterns: Iterable[str] = ("*",),
              exclude_dirs: tuple[str, ...] = ()) -> None:
    if not root.exists():
        plan.skipped.append(f"{root} (missing)")
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and any(p.match(pat) for pat in patterns) and not any(d in p.parts for d in exclude_dirs):
            _add(plan, repo, rtype, p, f"{prefix}/{p.relative_to(root).as_posix()}".lstrip("/"),
                 subst=rtype == "space" and p.suffix in SUBSTITUTE_SUFFIXES)


def plan_publish(user: str, root: Path = ROOT, data: Path = DATA, results: Path = RESULTS) -> Plan:
    plan = Plan(user=user)
    sft_repo, final_repo = f"{user}/tinyops-tiny-aya-sft", f"{user}/tinyops-tiny-aya"
    data_repo, space_repo = f"{user}/tinyops-data", f"{user}/tinyops-demo"
    # models: adapter folders are named in the selection files written by training
    for repo, sel_file, card in ((sft_repo, results / "runs" / "sft_selection.json", root / "hub" / "model_card_sft.md"),):
        if sel_file.exists():
            _add_tree(plan, repo, "model", Path(read_json(sel_file)["chosen"]["checkpoint"]), "")
        else:
            plan.skipped.append(f"{sft_repo}: {sel_file} missing (SFT not trained yet)")
        _add(plan, repo, "model", card, "README.md", subst=True)
    choice_f = results / "runs" / "final_model_choice.json"
    if choice_f.exists():
        final = read_json(choice_f)["final"]
        sel = results / "runs" / ("grpo_selection.json" if final == "sft+grpo" else "dpo_selection.json")
        src = Path(read_json(sel)["chosen"]["checkpoint"]) if final != "sft" and sel.exists() else None
        if src:
            _add_tree(plan, final_repo, "model", src, "")
        _add(plan, final_repo, "model", root / "hub" / "model_card_final.md", "README.md", subst=True)
    else:
        plan.skipped.append(f"{final_repo}: {choice_f} missing (final model not chosen yet)")
    # dataset
    for split in SPLITS:
        _add(plan, data_repo, "dataset", data / split / "scenarios.jsonl", f"scenarios/{split}.jsonl")
        for g in ("gold_episodes.jsonl", "gold_trajectories.jsonl"):
            _add(plan, data_repo, "dataset", data / split / g, f"gold/{split}/{g}")
    _add(plan, data_repo, "dataset", data / "TEST_HASHES.txt", "TEST_HASHES.txt")
    _add(plan, data_repo, "dataset", root / "hub" / "dataset_card.md", "README.md", subst=True)
    for sub in ("episodes", "metrics", "runs"):
        _add_tree(plan, data_repo, "dataset", results / sub, f"baselines/{sub}")
    _add_tree(plan, data_repo, "dataset", results / "final", "final", exclude_dirs=("plots",))
    _add_tree(plan, data_repo, "dataset", results / "train_logs", "training_logs")
    # space
    _add_tree(plan, space_repo, "space", root / "space", "", exclude_dirs=("__pycache__",))
    return plan


def scan_for_secrets(plan: Plan) -> list[str]:
    hits = []
    for it in plan.items:
        if it.local.suffix.lower() not in TEXT_SUFFIXES or it.local.stat().st_size > 50_000_000:
            continue
        text = it.local.read_text(encoding="utf-8", errors="ignore")
        hits += [f"{it.local}: {pat.pattern}" for pat in SECRET_PATTERNS if pat.search(text)]
    return hits


def execute(plan: Plan, api: Any = None, dry_run: bool = True) -> list[str]:
    """Create repos (exist_ok) and upload every file. Returns a log of the calls."""
    calls: list[str] = []
    secrets = scan_for_secrets(plan)
    if secrets:
        raise RuntimeError("possible secrets, refusing to upload:\n" + "\n".join(secrets))
    for (repo, rtype), items in plan.repos().items():
        calls.append(f"create_repo {repo} ({rtype})")
        if not dry_run:
            kw = {"space_sdk": "gradio"} if rtype == "space" else {}
            api.create_repo(repo, repo_type=rtype, exist_ok=True, **kw)
        for it in items:
            calls.append(f"upload {it.local} -> {repo}:{it.remote}")
            if not dry_run:
                body: Any = str(it.local)
                if it.subst:
                    body = it.local.read_text(encoding="utf-8").replace("<user>", plan.user).encode("utf-8")
                api.upload_file(path_or_fileobj=body, path_in_repo=it.remote, repo_id=repo, repo_type=rtype)
    return calls


app = typer.Typer(add_completion=False)


def _checked(user: str) -> str:
    if not user or "<" in user:
        raise typer.BadParameter("pass your real Hugging Face username with --user (the one you registered with)")
    return user


@app.command()
def plan(user: Annotated[str, typer.Option()]) -> None:
    p = plan_publish(_checked(user))
    for line in execute(p, dry_run=True):
        typer.echo(line)
    for s in p.skipped:
        typer.echo(f"SKIPPED {s}")


@app.command()
def push(user: Annotated[str, typer.Option()],
         do: Annotated[bool, typer.Option("--execute", help="really upload (default: dry run)")] = False) -> None:
    p = plan_publish(_checked(user))
    api = None
    if do:
        from huggingface_hub import HfApi

        api = HfApi()
    for line in execute(p, api, dry_run=not do):
        typer.echo(line)
    typer.echo("uploaded." if do else "dry run only; pass --execute to upload.")


if __name__ == "__main__":
    app()
