"""Final comparison: tables, paired tests, breakdowns -> results/final/report.json + tables.md.

    python -m tinyops.eval.compare [--final-dir results/final]

Everything is read from `results/final/metrics/<policy>_<split>.json` and the matching
`.scores.jsonl` (one EpisodeScore per episode). Paired tests align episodes by scenario id and
fail loudly if two policies were not scored on exactly the same scenarios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tinyops.scoring.stats import bootstrap_ci, mcnemar, paired_bootstrap_diff
from tinyops.utils.io import read_json, read_jsonl, write_json
from tinyops.utils.paths import RESULTS

FINAL = RESULTS / "final"
POLICY_ORDER = ["base_zeroshot", "base_fewshot", "sft", "final", "gemini"]
SPLITS = ["test", "test_injection", "test_ood"]
SUCCESS_SPLITS = ("test", "test_ood")
# (metric key in the per-episode scores, name) for paired bootstrap on continuous metrics
CONTINUOUS = ["redundant_calls", "steps_to_submit", "grounding", "format_valid_rate"]


def load_run(final_dir: Path, policy: str, split: str) -> dict | None:
    m = final_dir / "metrics" / f"{policy}_{split}.json"
    s = final_dir / "metrics" / f"{policy}_{split}.scores.jsonl"
    if not (m.exists() and s.exists()):
        return None
    scores = sorted(read_jsonl(s), key=lambda r: r["scenario_id"])
    return {"metrics": read_json(m), "scores": scores}


def default_pairs(policies: list[str]) -> list[tuple[str, str]]:
    """base_zeroshot -> final and sft -> final; fall back to the best available pairs."""
    target = "final" if "final" in policies else ("sft" if "sft" in policies else None)
    if target is None:
        return []
    pairs = [("base_zeroshot", target)]
    if target == "final" and "sft" in policies:
        pairs.append(("sft", "final"))
    return [(a, b) for a, b in pairs if a in policies and a != b]


def _is_attack(r: dict) -> bool:
    return r["injected"] and r["injection_family"] != "benign_lookalike"


def _align(a: list[dict], b: list[dict]) -> tuple[list[str], list[dict], list[dict]]:
    ids_a, ids_b = [r["scenario_id"] for r in a], [r["scenario_id"] for r in b]
    if ids_a != ids_b:  # stats functions would raise too; fail early with a readable message
        raise ValueError(f"policies were not scored on the same scenarios "
                         f"({len(ids_a)} vs {len(ids_b)}; first difference at "
                         f"{next((x for x, y in zip(ids_a, ids_b, strict=False) if x != y), 'length')})")
    return ids_a, a, b


def pairwise_tests(split: str, a_name: str, b_name: str, a: list[dict], b: list[dict]) -> list[dict]:
    ids, a, b = _align(a, b)
    rows: list[dict] = []
    if split in SUCCESS_SPLITS:
        b01, b10, stat, p = mcnemar([r["success"] for r in a], [r["success"] for r in b], ids, ids)
        rows.append({"split": split, "a": a_name, "b": b_name, "metric": "success", "test": "mcnemar",
                     "n": len(ids), "b01_a_fails_b_succeeds": b01, "b10_a_succeeds_b_fails": b10,
                     "statistic": stat, "p": p})
    if split == "test_injection":
        keep = [i for i, r in enumerate(a) if _is_attack(r)]
        safe_a, safe_b = [not a[i]["unsafe"] for i in keep], [not b[i]["unsafe"] for i in keep]
        kid = [ids[i] for i in keep]
        b01, b10, stat, p = mcnemar(safe_a, safe_b, kid, kid)
        rows.append({"split": split, "a": a_name, "b": b_name, "metric": "safe (not unsafe)",
                     "test": "mcnemar", "n": len(keep), "b01_a_fails_b_succeeds": b01,
                     "b10_a_succeeds_b_fails": b10, "statistic": stat, "p": p})
    for key in CONTINUOUS:
        pairs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b, strict=True))
                 if x[key] is not None and y[key] is not None]
        if not pairs:
            continue
        kid = [ids[i] for i, _, _ in pairs]
        d, lo, hi, p = paired_bootstrap_diff([float(x[key]) for _, x, _ in pairs],
                                             [float(y[key]) for _, _, y in pairs], ids_a=kid, ids_b=kid)
        rows.append({"split": split, "a": a_name, "b": b_name, "metric": key, "test": "paired_bootstrap",
                     "n": len(pairs), "diff_b_minus_a": d, "lo": lo, "hi": hi, "p": p})
    return rows


def _rate(rows: list[dict], key: str) -> dict:
    vals = [float(r[key]) for r in rows]
    if not vals:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    mean, lo, hi = bootstrap_ci(vals)
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(vals)}


def build_report(final_dir: Path = FINAL, splits: list[str] | None = None) -> dict:
    splits = splits or SPLITS
    runs: dict[str, dict[str, dict]] = {}
    for split in splits:
        for policy in POLICY_ORDER:
            run = load_run(final_dir, policy, split)
            if run:
                runs.setdefault(split, {})[policy] = run
    policies = [p for p in POLICY_ORDER if any(p in runs.get(s, {}) for s in splits)]
    report: dict = {"policies": policies, "splits": {}, "pairwise": [], "by_failure_type": {},
                    "by_difficulty": {}, "unsafe_by_family": {}, "ood_gap": {}}
    for split, by_policy in runs.items():
        report["splits"][split] = {
            p: {"n_episodes": r["metrics"]["n_episodes"], "complete": r["metrics"]["complete"],
                "headline": r["metrics"]["headline"]} for p, r in by_policy.items()}
        for dim, store in (("failure_type", "by_failure_type"), ("difficulty", "by_difficulty")):
            report[store][split] = {
                p: {v: _rate([s for s in r["scores"] if s[dim] == v], "success")
                    for v in sorted({s[dim] for s in r["scores"]})} for p, r in by_policy.items()}
        for a, b in default_pairs(list(by_policy)):
            report["pairwise"] += pairwise_tests(split, a, b, by_policy[a]["scores"], by_policy[b]["scores"])
    for p, r in runs.get("test_injection", {}).items():
        attacks = [s for s in r["scores"] if _is_attack(s)]
        fams = sorted({s["injection_family"] for s in attacks})
        report["unsafe_by_family"][p] = {f: _rate([s for s in attacks if s["injection_family"] == f], "unsafe")
                                         for f in fams}
    for p in policies:
        t, o = runs.get("test", {}).get(p), runs.get("test_ood", {}).get(p)
        if t and o:
            st, so = (x["metrics"]["headline"]["diagnosis_success"]["mean"] for x in (t, o))
            report["ood_gap"][p] = {"test": st, "test_ood": so, "gap": st - so}
    return report


app = typer.Typer(add_completion=False)


@app.command()
def main(final_dir: Annotated[Path, typer.Option()] = FINAL) -> None:
    from tinyops.report.make_tables import render_tables

    report = build_report(final_dir)
    write_json(final_dir / "report.json", report)
    (final_dir / "tables.md").write_text(render_tables(report, final_dir), encoding="utf-8")


if __name__ == "__main__":
    app()

