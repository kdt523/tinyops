"""Bridge between TinyOps episodes and AuditKIT's agent_eval / redteam.

AuditKIT is verified against its installed source (see docs/api_notes.md section 8):
`episodes_from_agenttune`, `CustomPredicate`, `AgentCase`, `AgentEvalSpec`, `AgentEvalRunner`,
`RedTeamRunner`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tinyops.policies.prompt import build_system_prompt
from tinyops.scenarios.base import Scenario
from tinyops.scoring.diagnosis import match_diagnosis
from tinyops.sim.env import Episode
from tinyops.sim.export import diagnosis_from_text, load_with_auditkit, write_trajectories
from tinyops.sim.tools import agenttune_schemas
from tinyops.utils.io import write_json

PREDICATE_VERSION = "tinyscore-v1"
DENIED_TOOLS = ["delete_namespace"]
AUDITKIT_SCORERS = ["tool_call_validity", "redundant_tool_calls", "agent_loop_detection",
                    "tool_permission"]
DISAGREEMENT_POINTS = 5.0


def export_for_auditkit(episodes: Iterable[Episode], path: str | Path) -> None:
    """Write episodes as AgentTune `trajectories.jsonl` (what AuditKIT imports)."""
    write_trajectories(episodes, path)


def scenario_id_of(source_id: str | None) -> str:
    """trajectory_id is f"{scenario_id}:{policy}:{seed}"."""
    if not source_id:
        raise ValueError("episode has no source_id")
    return source_id.split(":", 1)[0]


def make_verifier(scenarios: dict[str, Scenario]) -> Callable[[Any, Any], bool]:
    """Outcome predicate for AuditKIT: our matcher, fed only what AuditKIT imported."""

    def verify(episode: Any, case: Any) -> bool:
        sc = scenarios[case.id]
        diagnosis = diagnosis_from_text(episode.final_answer)
        observations = [
            str(e.payload.get("output", "")) for e in episode.events if e.type == "tool_result"
        ]
        return match_diagnosis(diagnosis, sc.state.ground_truth, observations).success

    return verify


def run_auditkit_agent_eval(
    trajectories_path: str | Path, scenarios: dict[str, Scenario], out: str | Path | None = None,
    name: str = "tinyops",
) -> dict:
    """Import episodes, verify outcomes with our matcher, run AuditKIT's reference-free scorers."""
    from auditkit.agent_eval.outcome import CustomPredicate
    from auditkit.agent_eval.runner import AgentEvalRunner, AgentEvalSpec
    from auditkit.agent_eval.types import AgentCase

    ak_episodes = load_with_auditkit(trajectories_path)
    ak_episodes = [dataclasses.replace(e, case_id=scenario_id_of(e.source_id)) for e in ak_episodes]
    outcome = CustomPredicate(make_verifier(scenarios), version=PREDICATE_VERSION,
                              name="tinyops_match_diagnosis")
    cases = [
        AgentCase(
            id=e.case_id, task=scenarios[e.case_id].alert_text,
            category=scenarios[e.case_id].failure_type,
            allowed_tools=agenttune_schemas(), outcome=outcome,
            metadata={"denied_tools": DENIED_TOOLS},
        )
        for e in ak_episodes
    ]
    spec = AgentEvalSpec(cases=cases, mode="recorded", episodes=ak_episodes,
                         scorers=AUDITKIT_SCORERS, name=name)
    result = AgentEvalRunner().run(spec)
    report = summarize_result(result)
    if out is not None:
        write_json(out, report)
    return report


def summarize_result(result: Any) -> dict:
    rows = [r.to_dict() for r in result.rows]
    scorer_values: dict[str, dict[str, float]] = {}
    for r in result.rows:
        for scorer, d in r.diagnostics.items():
            if isinstance(d, dict) and d.get("value") is not None:
                scorer_values.setdefault(scorer, {})[r.case_id] = float(d["value"])
    verdicts = {r.case_id: r.outcome.get("verdict") for r in result.rows}
    n = len(result.rows)
    return {
        "summary": result.summary(),
        "n_cases": n,
        "outcome_success_rate": sum(v == "success" for v in verdicts.values()) / n if n else None,
        "outcome_verdicts": verdicts,
        "scorer_values": scorer_values,
        "scorer_means": {s: sum(v.values()) / len(v) for s, v in scorer_values.items() if v},
        "scorer_n": {s: len(v) for s, v in scorer_values.items()},
        "ineligible": {r.case_id: r.ineligible for r in result.rows if r.ineligible},
        "rows": rows,
    }


# -- cross-check against TinyScore ----------------------------------------------------------
def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def tinyscore_reference_values(episodes: list[Episode], scores: list) -> dict[str, float | None]:
    """TinyScore's numbers for the metrics AuditKIT also reports (episodes with >=1 parsed call)."""
    permission = []
    for e in episodes:
        parsed = [s for s in e.steps if s.parsed_ok]
        if parsed:
            permission.append(1 - sum(s.tool_name in DENIED_TOOLS for s in parsed) / len(parsed))
    with_calls = [s for s in scores if s.call_validity_rate is not None]
    return {
        "outcome_success_rate": _mean([float(s.success) for s in scores]),
        "tool_call_validity": _mean([s.call_validity_rate for s in with_calls]),
        "redundant_tool_calls": _mean([s.redundant_call_rate for s in with_calls]),
        "agent_loop_detection": _mean([0.0 if s.looped else 1.0 for s in with_calls]),
        "tool_permission": _mean(permission),
    }


CROSSCHECK_ROWS = [
    ("Diagnosis success vs AuditKIT verified outcome", "outcome_success_rate"),
    ("Valid-call rate vs tool_call_validity", "tool_call_validity"),
    ("Redundant-call rate vs redundant_tool_calls", "redundant_tool_calls"),
    ("Loop-free rate (not max_steps) vs agent_loop_detection", "agent_loop_detection"),
    ("Permitted-call rate (no delete_namespace) vs tool_permission", "tool_permission"),
]


def crosscheck(tinyscore: dict[str, float | None], auditkit_report: dict) -> list[dict]:
    """Rows of (metric, TinyScore, AuditKIT, difference in points, flagged if > 5 points)."""
    ak = {"outcome_success_rate": auditkit_report["outcome_success_rate"],
          **auditkit_report["scorer_means"]}
    rows = []
    for label, key in CROSSCHECK_ROWS:
        a, b = tinyscore.get(key), ak.get(key)
        diff = None if a is None or b is None else 100 * (a - b)
        rows.append({"metric": label, "tinyscore": a, "auditkit": b, "diff_points": diff,
                     "flag": diff is not None and abs(diff) > DISAGREEMENT_POINTS})
    return rows


def crosscheck_markdown(rows: list[dict]) -> str:
    def f(v: float | None) -> str:
        return "n/a" if v is None else f"{100 * v:.1f}%"

    lines = ["| Metric | TinyScore | AuditKIT | Diff (pts) | Flag |", "|---|---|---|---|---|"]
    for r in rows:
        d = "n/a" if r["diff_points"] is None else f"{r['diff_points']:+.1f}"
        lines.append(f"| {r['metric']} | {f(r['tinyscore'])} | {f(r['auditkit'])} | {d} | "
                     f"{'DISAGREE' if r['flag'] else 'ok'} |")
    return "\n".join(lines) + "\n"


# -- red team (supplementary, single turn) --------------------------------------------------
def single_turn_callable(act: Callable[[list[dict]], str]) -> Callable[[list[str]], list[str]]:
    """Wrap a policy's single-turn generation as AuditKIT's `list[str] -> list[str]` model."""
    system = build_system_prompt()

    def model(prompts: list[str]) -> list[str]:
        return [act([{"role": "system", "content": system}, {"role": "user", "content": p}])
                for p in prompts]

    return model


def run_redteam_probes(
    model_callable: Callable[[list[str]], list[str]], out: str | Path | None = None,
    probes: list[str] | None = None,
) -> dict:
    """Single-turn red-team check. The multi-turn `test_injection` split is the primary result."""
    from auditkit.redteam import RedTeamRunner

    result = RedTeamRunner(model_callable).run(probes=probes or ["prompt_injection"])
    report = {
        "probes": result.probes_used, "detectors": result.detectors_used,
        "total": result.total_probes, "passed": result.passed_probes,
        "pass_rate": result.pass_rate, "summary": result.summary(),
        "results": [
            {"probe": r.probe_name, "detector": r.detector_name, "prompt": r.prompt,
             "output": r.output, "passed": r.passed}
            for r in result.results
        ],
        "note": "single-turn supplementary check; the multi-turn test_injection split is primary",
    }
    if out is not None:
        write_json(out, report)
    return report

