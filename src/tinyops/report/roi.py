"""Cost comparison: self-hosted TinyOps vs a closed API model, from measured numbers + cited prices.

    python -m tinyops.report.roi [--final-dir results/final] [--config configs/roi.yaml]

Measured (from results/final): tokens per episode, success and unsafe rates, episodes per GPU hour
(run manifests), training GPU hours (compute_log.csv). Cited (configs/roi.yaml): prices with a
source URL and date. A missing price or source raises; nothing is guessed.
"""

from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path
from typing import Annotated

import typer
import yaml

from tinyops.utils.io import read_json, read_jsonl, write_json
from tinyops.utils.paths import RESULTS

FINAL = RESULTS / "final"


def _need(section: dict, name: str, keys: list[str]) -> None:
    missing = [k for k in keys if section.get(k) in (None, "")]
    if missing:
        raise ValueError(f"configs/roi.yaml: {name} is missing {missing}. Every price needs a value, "
                         "a source_url and a date_checked.")
    if not str(section["source_url"]).startswith("http"):
        raise ValueError(f"configs/roi.yaml: {name}.source_url must be a URL")


def load_roi_config(path: Path | str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _need(cfg["closed_model"], "closed_model", ["price_per_1m_input_usd", "price_per_1m_output_usd",
                                                "source_url", "date_checked"])
    _need(cfg["gpu"], "gpu", ["price_per_hour_usd", "source_url", "date_checked"])
    for section in ("closed_model", "gpu"):  # YAML turns 2026-10-24 into a date; keep it a string
        cfg[section]["date_checked"] = str(cfg[section]["date_checked"])
    return cfg


def token_means(scores: list[dict]) -> tuple[float, float]:
    """Mean prompt and completion tokens per episode (episodes without usage are skipped)."""
    have = [s for s in scores if s["prompt_tokens"] is not None and s["completion_tokens"] is not None]
    if not have:
        raise ValueError("no token usage in the scores: model policies record it, scripted ones do not")
    return (sum(s["prompt_tokens"] for s in have) / len(have),
            sum(s["completion_tokens"] for s in have) / len(have))


def episodes_per_hour(manifests: list[dict]) -> float:
    """Total new episodes over total wall-clock hours of the given run manifests."""
    eps = sum(m["new_episodes_this_invocation"] for m in manifests)
    hours = sum(m["elapsed_s_this_invocation"] for m in manifests) / 3600
    if eps <= 0 or hours <= 0:
        raise ValueError("manifests contain no timed episodes (resumed runs report 0 new episodes)")
    return eps / hours


def training_hours(compute_log: Path, prefixes: list[str]) -> float:
    if not compute_log.exists():
        return 0.0
    rows = csv.DictReader(compute_log.open(encoding="utf-8"))
    return sum(float(r["hours"]) for r in rows if any(r["run"].startswith(p) for p in prefixes))


def break_even_volume(closed_per_episode: float, price_per_hour: float, eph: float,
                      hours_per_instance: float = 730.0) -> int | None:
    """Smallest monthly volume where the closed model costs at least the always-on instances."""
    if closed_per_episode <= price_per_hour / eph:
        return None  # at full utilisation self-hosting is never cheaper per episode
    cap = eph * hours_per_instance  # episodes per month per instance
    for n in range(1, 10_000):
        threshold = n * hours_per_instance * price_per_hour / closed_per_episode
        if threshold <= n * cap:
            return max(math.ceil(threshold), int((n - 1) * cap) + 1)
    return None


def compute_roi(cfg: dict, closed: dict, tiny: dict, train_gpu_hours: float) -> dict:
    """closed/tiny: {prompt_tokens, completion_tokens, success, unsafe}; tiny also episodes_per_hour."""
    cm, gpu = cfg["closed_model"], cfg["gpu"]
    per_ep_closed = (closed["prompt_tokens"] * cm["price_per_1m_input_usd"]
                     + closed["completion_tokens"] * cm["price_per_1m_output_usd"]) / 1e6
    price, eph, hpi = gpu["price_per_hour_usd"], tiny["episodes_per_hour"], cfg["always_on_hours_per_month"]
    train_cost = train_gpu_hours * price
    rows = []
    for v in cfg["volumes_per_month"]:
        closed_cost = v * per_ep_closed
        hours = math.ceil(v / eph)
        on_demand = hours * price
        always_on = max(1, math.ceil(hours / hpi)) * hpi * price
        self_host = max(on_demand, always_on)
        saving = closed_cost - self_host
        rows.append({
            "volume": v, "closed_cost_usd": closed_cost, "gpu_hours_needed": hours,
            "self_host_on_demand_usd": on_demand, "self_host_always_on_usd": always_on,
            "self_host_cost_usd": self_host, "monthly_saving_usd": saving,
            "months_to_recoup_training": (train_cost / saving) if saving > 0 else None,
            "closed_cost_per_success_usd": closed_cost / (v * closed["success"]) if closed["success"] else None,
            "self_host_cost_per_success_usd": self_host / (v * tiny["success"]) if tiny["success"] else None,
        })
    return {
        "inputs": {"closed": closed, "tiny": tiny, "training_gpu_hours": train_gpu_hours,
                   "prices": {"closed_model": cm, "gpu": gpu}},
        "closed_cost_per_episode_usd": per_ep_closed,
        "one_time_training_cost_usd": train_cost,
        "break_even_monthly_volume": break_even_volume(per_ep_closed, price, eph, hpi),
        "quality": {"success": {"closed": closed["success"], "tinyops": tiny["success"]},
                    "unsafe": {"closed": closed["unsafe"], "tinyops": tiny["unsafe"]}},
        "by_volume": rows,
    }


def render_markdown(roi: dict) -> str:
    cm = roi["inputs"]["prices"]["closed_model"]
    gpu = roi["inputs"]["prices"]["gpu"]
    q = roi["quality"]
    lines = [
        "# Cost comparison (monthly)", "",
        (f"Closed model `{cm['name']}`: ${cm['price_per_1m_input_usd']}/1M input, "
        f"${cm['price_per_1m_output_usd']}/1M output ({cm['source_url']}, checked {cm['date_checked']})."),
        f"GPU: {gpu['name']} at ${gpu['price_per_hour_usd']}/h ({gpu['source_url']}, checked {gpu['date_checked']}).",
        (f"Measured: {roi['inputs']['tiny']['episodes_per_hour']:.1f} TinyOps episodes per GPU-hour; one-time training "
        f"cost ${roi['one_time_training_cost_usd']:.2f} ({roi['inputs']['training_gpu_hours']:.2f} GPU-h)."), "",
        ("| Episodes / month | Closed model | TinyOps (self-host) | Saving | Months to recoup training | "
        "$ / successful diagnosis (closed) | $ / successful diagnosis (TinyOps) |"),
        "|---|---|---|---|---|---|---|",
    ]
    for r in roi["by_volume"]:
        months = "n/a" if r["months_to_recoup_training"] is None else f"{r['months_to_recoup_training']:.1f}"
        lines.append(f"| {r['volume']:,} | ${r['closed_cost_usd']:,.2f} | ${r['self_host_cost_usd']:,.2f} | "
                     f"${r['monthly_saving_usd']:,.2f} | {months} | ${r['closed_cost_per_success_usd']:.4f} | "
                     f"${r['self_host_cost_per_success_usd']:.4f} |")
    be = roi["break_even_monthly_volume"]
    lines += ["", (f"Break-even volume (closed cost reaches the always-on instance cost): "
                  f"{'never at full utilisation' if be is None else f'{be:,} episodes/month'}."), "",
              ("Quality next to cost: diagnosis success "
              f"{100 * q['success']['closed']:.1f}% (closed) vs {100 * q['success']['tinyops']:.1f}% (TinyOps); "
              f"unsafe rate on attacks {100 * q['unsafe']['closed']:.1f}% vs {100 * q['unsafe']['tinyops']:.1f}%."), ""]
    return "\n".join(lines)


def gather_inputs(final_dir: Path, closed_policy: str = "gemini", tiny_policy: str = "final") -> tuple[dict, dict]:
    def stats(policy: str) -> dict:
        scores = [s for sp in ("test", "test_injection", "test_ood")
                  for f in [final_dir / "metrics" / f"{policy}_{sp}.scores.jsonl"] if f.exists()
                  for s in read_jsonl(f)]
        p, c = token_means(scores)
        attacks = [s for s in scores if s["injected"] and s["injection_family"] != "benign_lookalike"]
        clean = [s for s in scores if not s["injected"]]
        return {"prompt_tokens": p, "completion_tokens": c,
                "success": sum(s["success"] for s in clean) / len(clean),
                "unsafe": sum(s["unsafe"] for s in attacks) / len(attacks) if attacks else 0.0}
    closed, tiny = stats(closed_policy), stats(tiny_policy)
    manifests = [read_json(p) for p in sorted((final_dir / "runs").glob(f"{tiny_policy}_*.json"))]
    tiny["episodes_per_hour"] = episodes_per_hour(manifests)
    return closed, tiny


app = typer.Typer(add_completion=False)


@app.command()
def main(final_dir: Annotated[Path, typer.Option()] = FINAL,
         config: Annotated[Path, typer.Option()] = Path("configs/roi.yaml"),
         compute_log: Annotated[Path, typer.Option()] = RESULTS / "compute_log.csv") -> None:
    cfg = load_roi_config(config)
    closed, tiny = gather_inputs(final_dir)
    roi = compute_roi(cfg, closed, tiny, training_hours(compute_log, cfg["training_run_prefixes"]))
    roi["generated_on"] = date.today().isoformat()
    write_json(final_dir / "roi.json", roi)
    (final_dir / "roi.md").write_text(render_markdown(roi), encoding="utf-8")
    typer.echo(render_markdown(roi))


if __name__ == "__main__":
    app()
