"""Stage B decision rules (pure functions on metrics): checkpoint choice, go/no-go, final model."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tinyops.utils.io import read_json, write_json

FORMAT_SLACK = 0.01      # a Stage B checkpoint may lose at most 1 point of format validity
MATCH_TOLERANCE = 0.02   # "matches SFT" = within 2 points of diagnosis success


def select_vs_reference(curve: list[dict], ref: dict) -> dict:
    """Best val success among checkpoints with unsafe <= reference and format >= reference - 1pt.

    `ref` is the SFT checkpoint's row on the same val subset. If no checkpoint qualifies SFT stays
    (`keep_reference` True, `chosen` None).
    """
    def unsafe(r: dict) -> float:
        return r["unsafe_rate"] or 0.0

    ref_unsafe, ref_format = unsafe(ref), ref["format_valid_rate"] or 0.0
    ok = [r for r in curve
          if unsafe(r) <= ref_unsafe + 1e-12 and (r["format_valid_rate"] or 0.0) >= ref_format - FORMAT_SLACK]
    chosen = min(ok, key=lambda r: (-r["diagnosis_success"], r["redundant_calls_per_ep"] or 0, r["step"])) if ok else None
    return {"chosen": chosen, "keep_reference": chosen is None, "reference": ref,
            "rule": "max val success s.t. unsafe <= SFT unsafe and format >= SFT format - 1pt",
            "candidates": curve}


def _m(metrics: dict, key: str) -> float | None:
    return metrics["headline"][key]["mean"]


def go_no_go(stage_b: dict, sft: dict) -> dict:
    """Is Stage B 'working'? (both args are full-val metrics files, e.g. results/metrics/*.json)

    Working = beats SFT on diagnosis success, or matches it (within 2 points) while using fewer
    redundant calls per episode; in both cases with no safety regression (unsafe not higher) and
    format validity within 1 point.
    """
    d_succ = _m(stage_b, "diagnosis_success") - _m(sft, "diagnosis_success")
    unsafe_b, unsafe_s = _m(stage_b, "unsafe_rate") or 0.0, _m(sft, "unsafe_rate") or 0.0
    safe = unsafe_b <= unsafe_s + 1e-12
    fmt_ok = _m(stage_b, "format_valid_rate") >= _m(sft, "format_valid_rate") - FORMAT_SLACK
    redundant_better = _m(stage_b, "redundant_calls_per_ep") < _m(sft, "redundant_calls_per_ep")
    beats = d_succ > 0
    matches_and_leaner = abs(d_succ) <= MATCH_TOLERANCE and redundant_better
    working = safe and fmt_ok and (beats or matches_and_leaner)
    why = ("beats SFT on diagnosis success" if beats else
           "matches SFT within 2 points with fewer redundant calls" if matches_and_leaner else
           "does not beat or efficiently match SFT")
    if not safe:
        why = "safety regression (unsafe rate above SFT)"
    elif not fmt_ok:
        why = "format validity dropped by more than 1 point"
    return {"working": working, "decision": "keep_stage_b" if working else "fallback_dpo_or_keep_sft",
            "reason": why, "diagnosis_success_delta": d_succ, "unsafe": {"stage_b": unsafe_b, "sft": unsafe_s}}


def choose_final(candidates: dict[str, dict], sft_name: str = "sft") -> dict:
    """Best of {sft, sft+grpo, sft+dpo} on full val: max success among those not less safe than SFT.

    Ties go to the simplest model (SFT first). Every candidate's numbers are returned.
    """
    base_unsafe = _m(candidates[sft_name], "unsafe_rate") or 0.0
    eligible = {n: m for n, m in candidates.items() if (_m(m, "unsafe_rate") or 0.0) <= base_unsafe + 1e-12}
    order = sorted(eligible, key=lambda n: (-_m(eligible[n], "diagnosis_success"), n != sft_name, n))
    return {"final": order[0], "eligible": sorted(eligible),
            "success": {n: _m(m, "diagnosis_success") for n, m in candidates.items()},
            "unsafe": {n: _m(m, "unsafe_rate") for n, m in candidates.items()},
            "rule": "max val success among candidates with unsafe <= SFT's; ties prefer SFT"}


def _cli() -> None:
    """`python -m tinyops.train.stage_b go-no-go|final ...` (reads/writes results/ files)."""
    app = typer.Typer(add_completion=False)

    @app.command("go-no-go")
    def cli_go_no_go(
        stage_b: Annotated[Path, typer.Option(help="full-val metrics of the selected Stage B model")],
        sft: Annotated[Path, typer.Option(help="full-val metrics of SFT")] = Path("results/metrics/sft_val.json"),
        out: Annotated[Path, typer.Option()] = Path("results/runs/stage_b_decision.json"),
    ) -> None:
        res = go_no_go(read_json(stage_b), read_json(sft))
        write_json(out, res)
        typer.echo(f"{res['decision']}: {res['reason']}")

    @app.command("final")
    def cli_final(
        candidate: Annotated[list[str], typer.Option(help="name=path/to/val_metrics.json, repeat")],
        out: Annotated[Path, typer.Option()] = Path("results/runs/final_model_choice.json"),
    ) -> None:
        cands = {name: read_json(path) for name, _, path in (c.partition("=") for c in candidate)}
        res = choose_final(cands)
        write_json(out, res)
        typer.echo(f"final model: {res['final']}")

    app()


if __name__ == "__main__":
    _cli()
