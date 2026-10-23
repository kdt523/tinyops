"""Figures from results/final/report.json and results/train_logs/ (PNG 200 dpi + SVG).

    python -m tinyops.report.make_plots [--final-dir results/final]

Consistent colours per policy (Okabe-Ito), labelled axes with units, titles that state split and n.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer

from tinyops.report.labels import POLICY_COLORS, POLICY_LABELS
from tinyops.utils.io import read_jsonl
from tinyops.utils.paths import RESULTS

FINAL = RESULTS / "final"
TRAIN_LOGS = RESULTS / "train_logs"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def _save(fig, stem: str, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / f"{stem}.png", outdir / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _stat(report: dict, split: str, policy: str, key: str) -> tuple[float, float, float, int] | None:
    s = report["splits"].get(split, {}).get(policy, {}).get("headline", {}).get(key)
    if not s or s["mean"] is None:
        return None
    return s["mean"], s["lo"], s["hi"], s["n"]


def plot_main_bars(report: dict, outdir: Path) -> list[Path]:
    """Success / format / schema validity on `test`, unsafe rate on `test_injection`, with CIs."""
    panels = [("test", "diagnosis_success", "Diagnosis success"), ("test", "format_valid_rate", "Format validity"),
              ("test", "schema_valid_rate", "Schema validity"),
              ("test_injection", "unsafe_rate", "Unsafe rate (attacks)")]
    pols = report["policies"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / max(1, len(pols))
    for j, pol in enumerate(pols):
        means, errs, xs = [], [[], []], []
        for i, (split, key, _) in enumerate(panels):
            st = _stat(report, split, pol, key)
            if st is None:
                continue
            xs.append(i + j * width)
            means.append(100 * st[0])
            errs[0].append(100 * (st[0] - st[1]))
            errs[1].append(100 * (st[2] - st[0]))
        ax.bar(xs, means, width, yerr=errs, capsize=2, color=POLICY_COLORS[pol],
               label=POLICY_LABELS.get(pol, pol))
    ax.set_xticks([i + width * (len(pols) - 1) / 2 for i in range(len(panels))])
    ax.set_xticklabels([f"{label}\n({split}, n={_n(report, split)})" for split, _, label in panels])
    ax.set_ylabel("Rate (%), 95% CI")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, ncol=min(5, len(pols)), loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_title("Before / after on the locked test splits", y=1.2)
    return _save(fig, "01_main_metrics", outdir)


def _n(report: dict, split: str) -> str:
    ns = {v["n_episodes"] for v in report["splits"].get(split, {}).values()}
    return "/".join(str(n) for n in sorted(ns)) if ns else "?"


def plot_heatmap(report: dict, outdir: Path, split: str = "test") -> list[Path]:
    data = report["by_failure_type"].get(split, {})
    pols = [p for p in report["policies"] if p in data]
    types = sorted({t for p in pols for t in data[p]})
    mat = np.array([[100 * (data[p].get(t, {}).get("mean") or 0.0) for p in pols] for t in types])
    fig, ax = plt.subplots(figsize=(1.4 * len(pols) + 3.5, 0.45 * len(types) + 1.5))
    im = ax.imshow(mat, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pols)), [POLICY_LABELS.get(p, p) for p in pols], rotation=20, ha="right")
    ax.set_yticks(range(len(types)), types)
    for i in range(len(types)):
        for j in range(len(pols)):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                    color="white" if mat[i, j] < 60 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Diagnosis success (%)")
    ax.set_title(f"Success by failure type ({split}, n={_n(report, split)})")
    return _save(fig, "02_success_by_failure_type", outdir)


def plot_unsafe_by_family(report: dict, outdir: Path) -> list[Path]:
    data = report["unsafe_by_family"]
    pols = [p for p in ("base_zeroshot", "final") if p in data] or list(data)
    fams = sorted({f for p in pols for f in data[p]})
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.8 / max(1, len(pols))
    for j, p in enumerate(pols):
        m = [100 * (data[p].get(f, {}).get("mean") or 0.0) for f in fams]
        lo = [100 * ((data[p].get(f, {}).get("mean") or 0.0) - (data[p].get(f, {}).get("lo") or 0.0)) for f in fams]
        hi = [100 * ((data[p].get(f, {}).get("hi") or 0.0) - (data[p].get(f, {}).get("mean") or 0.0)) for f in fams]
        ax.bar(np.arange(len(fams)) + j * width, m, width, yerr=[lo, hi], capsize=2,
               color=POLICY_COLORS[p], label=POLICY_LABELS.get(p, p))
    ax.set_xticks(np.arange(len(fams)) + width * (len(pols) - 1) / 2, fams, rotation=20, ha="right")
    ax.set_ylabel("Unsafe action rate (%)")
    ax.set_title(f"Obeying injected instructions, by attack family (test_injection, n={_n(report, 'test_injection')})")
    ax.legend(frameon=False)
    return _save(fig, "03_unsafe_by_family", outdir)


def plot_efficiency(report: dict, outdir: Path, split: str = "test") -> list[Path]:
    pols = [p for p in ("base_zeroshot", "final") if p in report["splits"].get(split, {})] or report["policies"]
    metrics = [("redundant_calls_per_ep", "Redundant calls per episode"), ("steps_to_submit", "Steps to submit")]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for ax, (key, label) in zip(axes, metrics, strict=True):
        for i, p in enumerate(pols):
            st = _stat(report, split, p, key)
            if st:
                ax.bar(i, st[0], yerr=[[st[0] - st[1]], [st[2] - st[0]]], capsize=3, color=POLICY_COLORS[p])
        ax.set_xticks(range(len(pols)), [POLICY_LABELS.get(p, p) for p in pols], rotation=15)
        ax.set_ylabel(f"{label} (count)")
    fig.suptitle(f"Efficiency ({split}, n={_n(report, split)})")
    return _save(fig, "04_efficiency", outdir)


def plot_training_curves(logs: Path, outdir: Path) -> list[Path]:
    """Val success vs step for SFT / Stage B, and mean reward per rollout batch."""
    import csv

    series: dict[str, tuple[list[int], list[float]]] = {}
    for name, f in (("SFT", "sft_val_curve.csv"), ("GRPO", "grpo_val_curve.csv"), ("DPO", "dpo_val_curve.csv")):
        p = logs / f
        if p.exists():
            rows = list(csv.DictReader(p.open(encoding="utf-8")))
            series[name] = ([int(r["step"]) for r in rows], [float(r["diagnosis_success"]) for r in rows])
    rewards = logs / "grpo_rewards.jsonl"
    if not series and not rewards.exists():
        return []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    colors = {"SFT": POLICY_COLORS["sft"], "GRPO": POLICY_COLORS["final"], "DPO": POLICY_COLORS["base_fewshot"]}
    for name, (x, y) in series.items():
        axes[0].plot(x, [100 * v for v in y], marker="o", color=colors[name], label=name)
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Val diagnosis success (%), val subset")
    axes[0].legend(frameon=False)
    if rewards.exists():
        rows = read_jsonl(rewards)
        tot = np.array([r["total"] for r in rows])
        k = 8
        n = len(tot) // k
        if n:
            axes[1].plot(range(n), tot[: n * k].reshape(n, k).mean(axis=1), color=POLICY_COLORS["final"])
    axes[1].set_xlabel("Rollout batch (8 rollouts)")
    axes[1].set_ylabel("Mean combined reward")
    fig.suptitle("Training curves (validation subset and GRPO reward)")
    return _save(fig, "05_training_curves", outdir)


def make_all(final_dir: Path = FINAL, logs: Path = TRAIN_LOGS) -> list[Path]:
    report = json.loads((final_dir / "report.json").read_text(encoding="utf-8"))
    out = final_dir / "plots"
    paths = plot_main_bars(report, out) + plot_heatmap(report, out)
    if report["unsafe_by_family"]:
        paths += plot_unsafe_by_family(report, out)
    paths += plot_efficiency(report, out) + plot_training_curves(logs, out)
    return paths


app = typer.Typer(add_completion=False)


@app.command()
def main(final_dir: Annotated[Path, typer.Option()] = FINAL,
         logs: Annotated[Path, typer.Option()] = TRAIN_LOGS) -> None:
    for p in make_all(final_dir, logs):
        typer.echo(str(p))


if __name__ == "__main__":
    app()
