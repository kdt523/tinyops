"""Pre-submission checklist: everything that can be verified automatically, plus the manual items.

    python -m tinyops.report.check_submission      # exit code 1 if any check FAILs
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import typer

from tinyops.report.publish import SECRET_PATTERNS
from tinyops.scenarios.build_dataset import verify_test_hashes
from tinyops.utils.io import sha256_file
from tinyops.utils.paths import DATA, RESULTS, ROOT

PASS, FAIL, WARN, MANUAL = "PASS", "FAIL", "WARN", "MANUAL"
REQUIRED = [
    "README.md", "LICENSE", "docs/ARTICLE.md", "docs/DEMO_SCRIPT.md", "docs/LEXSI_FEEDBACK.md", "docs/METRICS.md",
    "docs/DECISIONS.md", "docs/BASELINE_FINDINGS.md", "docs/FAILURE_ANALYSIS.md", "hub/model_card_final.md",
    "hub/dataset_card.md", "space/app.py", "space/requirements.txt", "space/README.md", "space/report.json",
    "space/replays/index.json", "data/TEST_HASHES.txt", "results/final/report.json", "results/final/tables.md",
    "results/final/roi.json", "results/final/FINAL_RUN.json",
]
# placeholders that must be gone before publishing; <user> is allowed in hub/ and space/ (substituted on upload)
LEFTOVERS = [r"<github-user>", r"<author>", r"YOUR_[A-Z_]+", r"⟨pending", r"_Pending", r"TODO\(human\)"]
PUBLISH_DOCS = ["README.md", "docs/ARTICLE.md", "hub/model_card_final.md", "hub/dataset_card.md", "space/README.md"]
SCAN_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".csv", ".ipynb", ".cfg"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".ruff_cache", ".pytest_cache", "episodes", "node_modules"}
MANUAL_ITEMS = [
    "Repo is public and CI is green on GitHub",
    "Model(s), dataset and Space are public on Hugging Face and the Space loads in replay mode",
    "HF Article is published (paste docs/ARTICLE.md; upload the figures from results/final/plots/)",
    "All links cross-checked: article <-> Space <-> model <-> dataset <-> GitHub",
    "HF usernames in the submission match the registration form (credits are matched on them)",
    "Submission form filled; confirmation screenshot saved in docs/",
    "Demo video recorded, uploaded unlisted, linked from the Space and the article",
    "Posted on LinkedIn/X tagging Lexsi Labs, Cohere Labs and Hugging Face; shared in Discord",
    "Confirmed the Gemini price in configs/roi.yaml against the official pricing page",
    "Hackathon rules respected: code written Nov 1-14, a Cohere model used, artifacts hosted on Hugging Face",
]


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


def _text(root: Path, rel: str) -> str | None:
    p = root / rel
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else None


def check_files(root: Path) -> Check:
    missing = [r for r in REQUIRED if not (root / r).exists()]
    return Check("required files exist", FAIL if missing else PASS, ", ".join(missing))


def check_hashes(root: Path, data: Path) -> Check:
    try:
        verify_test_hashes(data)
        return Check("locked test splits match TEST_HASHES.txt", PASS)
    except (FileNotFoundError, ValueError) as e:
        return Check("locked test splits match TEST_HASHES.txt", FAIL, str(e))


def check_final_lock(results: Path) -> Check:
    f = results / "final" / "FINAL_RUN.json"
    if not f.exists():
        return Check("final run is locked (FINAL_RUN.json)", FAIL, "missing: run skill 8")
    rec = json.loads(f.read_text(encoding="utf-8"))
    if rec.get("missing"):
        return Check("final run is locked (FINAL_RUN.json)", FAIL, f"incomplete: {rec['missing']}")
    bad = [rel for rel, digest in rec.get("episodes_sha256", {}).items()
           if not (results / "final" / rel).exists() or sha256_file(results / "final" / rel) != digest]
    return Check("final run is locked (FINAL_RUN.json)", FAIL if bad else PASS,
                 f"episode files changed since the lock: {bad}" if bad else "")


def check_leftovers(root: Path) -> Check:
    hits = []
    for rel in PUBLISH_DOCS:
        t = _text(root, rel)
        if t is None:
            continue
        pats = list(LEFTOVERS) + ([] if rel.startswith(("hub/", "space/")) else [r"<user>"])
        hits += [f"{rel}: {m.group(0)}" for p in pats for m in re.finditer(p, t)][:5]
    t = _text(root, "LICENSE") or ""
    hits += ["LICENSE: <author>"] if "<author>" in t else []
    return Check("no leftover placeholders / pending markers", FAIL if hits else PASS, "; ".join(hits[:8]))


def check_before_after(results: Path) -> Check:
    f = results / "final" / "report.json"
    if not f.exists():
        return Check(">=2 before/after metrics with CIs", FAIL, "report.json missing")
    rep = json.loads(f.read_text(encoding="utf-8"))
    ok = []
    for split, key in (("test", "diagnosis_success"), ("test_injection", "unsafe_rate"), ("test", "format_valid_rate")):
        a = rep["splits"].get(split, {}).get("base_zeroshot", {}).get("headline", {}).get(key)
        b = rep["splits"].get(split, {}).get("final", {}).get("headline", {}).get(key)
        if a and b and a["lo"] is not None and b["lo"] is not None:
            ok.append(key)
    return Check(">=2 before/after metrics with CIs", PASS if len(ok) >= 2 else FAIL, f"found: {ok}")


def check_failure_doc(root: Path) -> Check:
    t = _text(root, "docs/FAILURE_ANALYSIS.md") or ""
    if "STATUS: CONFIRMED" in t:
        return Check("failure analysis confirmed by a human", PASS)
    return Check("failure analysis confirmed by a human", FAIL, "status is DRAFT or the file is missing")


def check_readme_results(root: Path) -> Check:
    t = _text(root, "README.md") or ""
    m = re.search(r"<!-- RESULTS:START -->(.*?)<!-- RESULTS:END -->", t, re.DOTALL)
    if not m:
        return Check("README results block", FAIL, "markers missing")
    return Check("README results block", FAIL if "Pending" in m.group(1) or "|" not in m.group(1) else PASS,
                 "still pending: run make_tables --update-readme" if "|" not in m.group(1) else "")


def check_roi(root: Path) -> Check:
    t = _text(root, "docs/ARTICLE.md") or ""
    return Check("ROI section in the article (optional, preferred)", PASS if "| Episodes / month |" in t else WARN,
                 "" if "| Episodes / month |" in t else "no cost table in the article")


def check_replays(root: Path) -> Check:
    idx = root / "space" / "replays" / "index.json"
    if not idx.exists():
        return Check("Space replays bundled", FAIL, "space/replays/index.json missing")
    n = json.loads(idx.read_text(encoding="utf-8"))["n"]
    spec = importlib.util.spec_from_file_location("replay_view", root / "space" / "replay_view.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("replay_view", mod)
    spec.loader.exec_module(mod)
    reps = mod.load_replays(root / "space" / "replays")
    footer = mod.footer_md(root / "space" / "report.json")
    bad = []
    if len(reps) != n or n < 10:
        bad.append(f"{len(reps)} replays loaded, index says {n} (need >= 10)")
    if "pending" in footer:
        bad.append("footer table pending")
    return Check("Space replay mode has data", FAIL if bad else PASS, "; ".join(bad))


def check_secrets(root: Path) -> Check:
    hits = []
    for p in root.rglob("*"):
        if (p.is_file() and p.suffix in SCAN_SUFFIXES and not (set(p.relative_to(root).parts) & SKIP_DIRS)
                and p.stat().st_size < 5_000_000 and p.name != ".env.example"):
            t = p.read_text(encoding="utf-8", errors="ignore")
            hits += [f"{p.relative_to(root)}" for pat in SECRET_PATTERNS if pat.search(t)]
    gi = _text(root, ".gitignore") or ""
    if ".env" not in gi:
        hits.append(".gitignore does not ignore .env")
    return Check("no secrets in the repository", FAIL if hits else PASS, "; ".join(sorted(set(hits))[:6]))


def run_checks(root: Path = ROOT, results: Path = RESULTS, data: Path = DATA) -> list[Check]:
    checks = [check_files(root), check_hashes(root, data), check_final_lock(results), check_leftovers(root),
              check_before_after(results), check_failure_doc(root), check_readme_results(root),
              check_roi(root), check_replays(root), check_secrets(root)]
    checks += [Check(item, MANUAL) for item in MANUAL_ITEMS]
    return checks


def render(checks: list[Check]) -> str:
    icon = {PASS: "[PASS]  ", FAIL: "[FAIL]  ", WARN: "[WARN]  ", MANUAL: "[ ] you "}
    return "\n".join(f"{icon[c.status]}{c.name}" + (f"  — {c.detail}" if c.detail else "") for c in checks)


app = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    checks = run_checks()
    typer.echo(render(checks))
    n_fail = sum(c.status == FAIL for c in checks)
    typer.echo(f"\n{n_fail} automatic check(s) failing; {sum(c.status == MANUAL for c in checks)} manual items remain.")
    raise typer.Exit(1 if n_fail else 0)


if __name__ == "__main__":
    app()
