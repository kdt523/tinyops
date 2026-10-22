"""Export episodes in AgentTune's trajectories.jsonl shape and load them through AuditKIT."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tinyops.policies.base import format_tool_call_text
from tinyops.policies.prompt import build_system_prompt, format_tool_result
from tinyops.sim.env import Episode, StepRecord
from tinyops.sim.tools import ToolCall
from tinyops.utils.io import write_jsonl


def step_text(step: StepRecord) -> str:
    """What the policy said at this step (reconstructed from the call if no raw text)."""
    if step.raw_model_text is not None:
        return step.raw_model_text
    if step.tool_name is not None:
        return format_tool_call_text(ToolCall(name=step.tool_name, arguments=step.arguments or {}))
    return ""


def conversation_from_episode(ep: Episode) -> list[dict]:
    """The chat exactly as the policy saw it: system, alert, then assistant/user turns."""
    msgs: list[dict] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": ep.alert_text},
    ]
    for i, s in enumerate(ep.steps):
        msgs.append({"role": "assistant", "content": step_text(s)})
        # the env ends the episode on the last step, so no result turn follows it
        # (except after a policy_error, where the last step's result was already fed back)
        last = i == len(ep.steps) - 1
        if not last or ep.termination == "policy_error":
            msgs.append(
                {"role": "user", "content": format_tool_result(s.tool_name or "parser", s.observation)}
            )
    return msgs


def diagnosis_text(diagnosis: dict | None) -> str:
    if not diagnosis:
        return ""
    ev = "\n".join(f"- {e}" for e in diagnosis.get("evidence", []))
    return (
        f"{diagnosis.get('root_cause', '')}\nEvidence:\n{ev}\n"
        f"Recommended fix: {diagnosis.get('recommended_fix', '')}"
    )


def diagnosis_from_text(text: str | None) -> dict | None:
    """Inverse of `diagnosis_text` (used when only AgentTune's final_response is available)."""
    if not text or not text.strip():
        return None
    head, _, rest = text.partition("\nEvidence:\n")
    evidence_part, _, fix = rest.partition("\nRecommended fix:")
    evidence = [ln[2:] for ln in evidence_part.splitlines() if ln.startswith("- ")]
    return {"root_cause": head.strip(), "evidence": evidence, "recommended_fix": fix.strip()}


def _call_id(step: StepRecord) -> str:
    return f"call_{step.index}"


def episode_to_trajectory_line(ep: Episode) -> dict[str, Any]:
    steps = []
    for s in ep.steps:
        calls = (
            [{"id": _call_id(s), "name": s.tool_name, "arguments": s.arguments or {}}]
            if s.parsed_ok and s.tool_name
            else []
        )
        steps.append(
            {
                "step_number": s.index,
                "state": f"[turn={s.index}]",
                "action": {"tool_calls": calls},
                "observation": s.observation,
                # None on purpose: AuditKIT's agent_loop_detection treats `thought` as reasoning
                # text and flags consecutive near-identical tool-call JSON as "stagnation"
                # (false loops). The raw model text is kept in metadata.conversation.
                "thought": None,
                "reward": None,
            }
        )
    return {
        "task": ep.alert_text,
        "steps": steps,
        "reward": None,
        "trajectory_id": f"{ep.scenario_id}:{ep.policy_name}:{ep.seed}",
        "final_response": diagnosis_text(ep.diagnosis),
        "metadata": {
            "scenario_id": ep.scenario_id,
            "policy": ep.policy_name,
            "termination": ep.termination,
            "prompt_version": ep.meta.get("prompt_version"),
            "tool_call_count": sum(1 for s in ep.steps if s.parsed_ok),
            "conversation": conversation_from_episode(ep),
        },
    }


def write_trajectories(episodes: Iterable[Episode], path: str | Path) -> None:
    write_jsonl(path, (episode_to_trajectory_line(e) for e in episodes))


def episode_to_eventlog(ep: Episode):
    """Light-tier AgentTune EventLog (EventLog's constructor is public). Needs agenttune."""
    from agenttune.agentic.events import Event, EventKind, EventLog

    log = EventLog(tier="light")
    for s in ep.steps:
        calls = (
            [{"id": _call_id(s), "name": s.tool_name, "arguments": s.arguments or {}}]
            if s.parsed_ok and s.tool_name
            else []
        )
        log.append(Event(EventKind.TOOL_CALL, {"action": {"tool_calls": calls}}))
        log.append(Event(EventKind.TOOL_RESULT, {"output": s.observation, "tool_call_id": _call_id(s)}))
        log.append(Event(EventKind.TURN_COMPLETE, {}))
    log.append(Event(EventKind.TEXT, {"text": diagnosis_text(ep.diagnosis)}))
    return log


def load_with_auditkit(path: str | Path):
    from auditkit.agent_eval import episodes_from_agenttune

    return episodes_from_agenttune(str(path))
