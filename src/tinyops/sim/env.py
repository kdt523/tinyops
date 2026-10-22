"""KubeSim episode environment: step semantics, step records, episode runner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from jsonschema import Draft202012Validator
from pydantic import BaseModel

from tinyops.sim.cluster import ClusterState
from tinyops.sim.safety import evaluate_unknown_tool
from tinyops.sim.tools import (
    TERMINAL_TOOL,
    TOOL_NAMES,
    TOOL_SCHEMAS,
    WRITE_TOOLS,
    ToolBox,
    ToolCall,
)

if TYPE_CHECKING:
    from tinyops.policies.base import Policy

PARSE_ERROR_OBSERVATION = (
    "Error: could not parse a tool call. Respond with exactly one tool call in the required format."
)
_VALIDATORS = {n: Draft202012Validator(s) for n, s in TOOL_SCHEMAS.items()}


class StepRecord(BaseModel):
    index: int
    raw_model_text: str | None
    parsed_ok: bool
    tool_name: str | None
    arguments: dict | None
    tool_exists: bool
    schema_valid: bool
    schema_error: str | None
    is_duplicate: bool
    is_write: bool
    is_unsafe: bool
    is_unnecessary_write: bool
    is_correct_fix: bool
    observation: str
    latency_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class Episode(BaseModel):
    scenario_id: str
    policy_name: str
    seed: int
    alert_text: str
    steps: list[StepRecord]
    diagnosis: dict | None
    termination: Literal["submitted", "max_steps", "parse_failures", "policy_error"]
    tool_outputs_seen: list[str]
    meta: dict = {}


def canonical_args(arguments: dict) -> str:
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False)


def _first_schema_error(tool: str, arguments: dict) -> str | None:
    errors = sorted(_VALIDATORS[tool].iter_errors(arguments), key=lambda e: list(map(str, e.path)))
    return errors[0].message if errors else None


class KubeSimEnv:
    """One episode over one ClusterState. A turn is one policy output, parsed or not."""

    def __init__(
        self,
        state: ClusterState,
        max_steps: int = 8,
        max_parse_failures: int = 3,
        alert_text: str = "",
    ) -> None:
        self.state = state
        self.max_steps = max_steps
        self.max_parse_failures = max_parse_failures
        self.alert_text = alert_text
        self.toolbox = ToolBox(state)
        self.reset()

    def reset(self) -> str:
        self.toolbox = ToolBox(self.state)
        self.steps: list[StepRecord] = []
        self.observations: list[str] = []
        self.diagnosis: dict | None = None
        self.termination: str | None = None
        self._seen: set[tuple[str, str]] = set()
        self._consecutive_parse_failures = 0
        return self.alert_text

    @property
    def done(self) -> bool:
        return self.termination is not None

    def step(
        self, call: ToolCall | None, raw_text: str | None = None
    ) -> tuple[str, bool, StepRecord]:
        if self.done:
            raise RuntimeError("episode already finished; call reset()")
        idx = len(self.steps)
        flags = {
            "is_write": False, "is_unsafe": False,
            "is_unnecessary_write": False, "is_correct_fix": False,
        }
        if call is None:
            self._consecutive_parse_failures += 1
            rec = StepRecord(
                index=idx, raw_model_text=raw_text, parsed_ok=False, tool_name=None,
                arguments=None, tool_exists=False, schema_valid=False,
                schema_error="could not parse a tool call", is_duplicate=False,
                observation=PARSE_ERROR_OBSERVATION, **flags,
            )
        else:
            self._consecutive_parse_failures = 0
            rec = self._execute(idx, call, raw_text, flags)
        self.steps.append(rec)
        self.observations.append(rec.observation)
        self._maybe_finish(rec)
        return rec.observation, self.done, rec

    def _execute(self, idx: int, call: ToolCall, raw_text: str | None, flags: dict) -> StepRecord:
        base = {
            "index": idx, "raw_model_text": raw_text, "parsed_ok": True,
            "tool_name": call.name, "arguments": call.arguments,
        }
        if call.name not in TOOL_SCHEMAS:
            flags["is_unsafe"] = evaluate_unknown_tool(call.name, self.state.ground_truth)
            obs = f'Error: unknown tool "{call.name}". Available tools: {", ".join(TOOL_NAMES)}'
            return StepRecord(
                **base, tool_exists=False, schema_valid=False, schema_error="unknown tool",
                is_duplicate=False, observation=obs, **flags,
            )
        key = (call.name, canonical_args(call.arguments))
        is_dup = key in self._seen
        self._seen.add(key)
        err = _first_schema_error(call.name, call.arguments)
        if err is not None:
            return StepRecord(
                **base, tool_exists=True, schema_valid=False, schema_error=err,
                is_duplicate=is_dup,
                observation=f"Error: invalid arguments for {call.name}: {err}", **flags,
            )
        obs, verdict = self.toolbox.dispatch(call.name, call.arguments)
        if verdict is not None and call.name in WRITE_TOOLS:
            flags = {
                "is_write": verdict.is_write, "is_unsafe": verdict.is_unsafe,
                "is_unnecessary_write": verdict.is_unnecessary_write,
                "is_correct_fix": verdict.is_correct_fix,
            }
        if call.name == TERMINAL_TOOL:
            self.diagnosis = dict(call.arguments)
        return StepRecord(
            **base, tool_exists=True, schema_valid=True, schema_error=None,
            is_duplicate=is_dup, observation=obs, **flags,
        )

    def _maybe_finish(self, rec: StepRecord) -> None:
        if self.diagnosis is not None:
            self.termination = "submitted"
        elif self._consecutive_parse_failures >= self.max_parse_failures:
            self.termination = "parse_failures"
        elif len(self.steps) >= self.max_steps:
            self.termination = "max_steps"


def run_episode(
    state: ClusterState,
    policy: Policy,
    alert_text: str,
    max_steps: int = 8,
    seed: int = 1234,
    scenario_id: str = "adhoc",
) -> Episode:
    """Loop: build messages -> policy.act -> env.step -> append observation."""
    from tinyops.policies.prompt import PROMPT_VERSION, build_system_prompt, format_tool_result

    env = KubeSimEnv(state, max_steps=max_steps, alert_text=alert_text)
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": env.reset()},
    ]
    termination = None
    while not env.done:
        try:
            out = policy.act(messages)
        except Exception as e:  # noqa: BLE001 - a crashing policy ends the episode, not the run
            termination = "policy_error"
            error = f"{type(e).__name__}: {e}"
            break
        messages.append({"role": "assistant", "content": out.text})
        obs, done, rec = env.step(out.tool_call, out.text)
        rec.latency_s = out.usage.get("latency_s")
        rec.prompt_tokens = out.usage.get("prompt_tokens")
        rec.completion_tokens = out.usage.get("completion_tokens")
        if not done:
            messages.append(
                {"role": "user", "content": format_tool_result(rec.tool_name or "parser", obs)}
            )
    meta = {"prompt_version": PROMPT_VERSION, "max_steps": max_steps}
    if termination == "policy_error":
        meta["policy_error"] = error
    return Episode(
        scenario_id=scenario_id,
        policy_name=getattr(policy, "name", type(policy).__name__),
        seed=seed,
        alert_text=alert_text,
        steps=env.steps,
        diagnosis=env.diagnosis,
        termination=termination or env.termination,
        tool_outputs_seen=list(env.observations),
        meta=meta,
    )
