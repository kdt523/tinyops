"""Rendering helpers for the Space (replay mode). Pure Python: no tinyops/torch import needed."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path

TOOL_RESULT_CHARS = 400
BADGE_OK, BADGE_WARN, BADGE_BAD = "✅", "⚠️", "❌"


def load_replays(replay_dir: Path | str) -> dict[str, dict]:
    out = {}
    for p in sorted(Path(replay_dir).glob("*.json")):
        if p.name == "index.json":
            continue
        rep = json.loads(p.read_text(encoding="utf-8"))
        out[rep["scenario_id"]] = rep
    return out


def choices(replays: dict[str, dict], attacks_only: bool | None = None) -> list[tuple[str, str]]:
    """(label, scenario_id) for the dropdown. attacks_only: True = injected only, False = clean only."""
    rows = []
    for sid, r in replays.items():
        if attacks_only is not None and bool(r["injected"]) != attacks_only:
            continue
        tag = f" + {r['injection_family']}" if r["injected"] else ""
        rows.append((f"{r['failure_type']} ({r['difficulty']}){tag} [{r['split']}]", sid))
    return sorted(rows)


def step_badge(step: dict) -> tuple[str, str]:
    """(badge, label) for one step record."""
    if step.get("is_unsafe"):
        return BADGE_BAD, "unsafe action"
    if not step.get("parsed_ok"):
        return BADGE_WARN, "could not parse a tool call"
    if not step.get("tool_exists"):
        return BADGE_WARN, "unknown tool"
    if not step.get("schema_valid"):
        return BADGE_WARN, "invalid arguments"
    return BADGE_OK, "valid"


def _clip(text: str, n: int = TOOL_RESULT_CHARS) -> str:
    return text if len(text) <= n else text[:n] + " …"


def transcript_md(run: dict, clip: int = TOOL_RESULT_CHARS) -> str:
    ep = run["episode"]
    out = [f"**{run['label']}**", ""]
    for s in ep["steps"]:
        badge, label = step_badge(s)
        out.append(f"{badge} **step {s['index'] + 1}** ({label})")
        out.append("```json\n" + (s.get("raw_model_text") or "").strip() + "\n```")
        out.append("```\n" + _clip(s["observation"], clip) + "\n```")
    return "\n".join(out)


def verdict(run: dict) -> dict:
    sc, ep = run["score"], run["episode"]
    return {
        "correct": bool(sc["success"]),
        "safe": not sc["unsafe"],
        "steps": len(ep["steps"]),
        "termination": ep["termination"],
        "root_cause": (ep.get("diagnosis") or {}).get("root_cause"),
        "evidence": (ep.get("diagnosis") or {}).get("evidence") or [],
        "recommended_fix": (ep.get("diagnosis") or {}).get("recommended_fix"),
    }


def verdict_md(run: dict) -> str:
    v = verdict(run)
    safety = "🔴 unsafe attempt blocked" if not v["safe"] else "🟢 no unsafe action"
    lines = [(f"**{run['label']}**: {'✅ correct diagnosis' if v['correct'] else '❌ wrong or missing diagnosis'} · {safety} · "
             f"{v['steps']} steps · ended: `{v['termination']}`"), ""]
    if v["root_cause"]:
        lines += [f"**Root cause:** {v['root_cause']}", "", "**Evidence:**"] + [f"- `{e}`" for e in v["evidence"]]
        lines += ["", f"**Recommended fix:** {v['recommended_fix']}"]
    else:
        lines.append("_No diagnosis submitted._")
    return "\n".join(lines)


def scenario_md(rep: dict) -> str:
    lines = [f"**Alert:** {rep['alert']}", "",
             f"`{rep['failure_type']}` · {rep['difficulty']} · split `{rep['split']}`"]
    if rep["injected"]:
        lines += ["", (f"⚠️ **Injected attack** (`{rep['injection_family']}` in the {rep['injection_location']}): "
                      f"`{rep['injected_payload']}`")]
    lines += ["", f"_Why this replay: {rep['bucket_name']}._"]
    return "\n".join(lines)


def footer_md(report_path: Path | str) -> str:
    """Headline table from the copied results/final/report.json (nothing is typed here)."""
    p = Path(report_path)
    if not p.exists():
        return "_Results table pending: `report.json` has not been copied into the Space yet._"
    report = json.loads(p.read_text(encoding="utf-8"))
    names = {"base_zeroshot": "Base Tiny Aya", "base_fewshot": "Base few-shot", "sft": "SFT", "final": "TinyOps",
             "gemini": "Gemini"}
    cols = [("test", "diagnosis_success", "Success (test)"), ("test_injection", "unsafe_rate", "Unsafe rate (injection)"),
            ("test_ood", "diagnosis_success", "Success (OOD)")]

    def cell(policy: str, split: str, key: str) -> str:
        st = report["splits"].get(split, {}).get(policy, {}).get("headline", {}).get(key)
        if not st or st["mean"] is None:
            return "n/a"
        return f"{100 * st['mean']:.1f}% [{100 * st['lo']:.1f}, {100 * st['hi']:.1f}]"

    rows = ["| Model | " + " | ".join(c[2] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for p_ in report["policies"]:
        rows.append(f"| {names.get(p_, p_)} | " + " | ".join(cell(p_, s, k) for s, k, _ in cols) + " |")
    return "\n".join(rows) + "\n\n_95% bootstrap CIs; locked test splits; see the model card and article for the full tables._"


def play(rep: dict, model_name: str, delay: float = 0.0, sleep: Callable[[float], None] = time.sleep) -> Iterator[tuple[str, str, str]]:
    """Stream a replay step by step: yields (scenario_md, transcript_md, verdict_md)."""
    run = rep["runs"]["base" if model_name == "Base Tiny Aya" else "final"]
    steps = run["episode"]["steps"]
    for i in range(len(steps)):
        partial = {**run, "episode": {**run["episode"], "steps": steps[: i + 1]}}
        yield scenario_md(rep), transcript_md(partial), ""
        sleep(delay)
    yield scenario_md(rep), transcript_md(run), verdict_md(run)
