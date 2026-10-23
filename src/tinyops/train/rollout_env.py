"""Per-rollout scenario binding for AgentTune's tool-calling rollouts.

AgentTune calls tools as `fn(**model_arguments)`: no dataset column reaches them
(docs/api_notes.md section 2). Rollouts of one trainer run one prompt at a time, and
`pre_step_hook(0, conversation, tools)` fires at the start of every trajectory in the same thread
that later runs the tools. So the hook identifies the scenario from the alert in the conversation
(alert texts are unique in the RL dataset), and the tools read the bound environment from a
ContextVar (with a module-level fallback for tools that run in a thread with an empty context).
Tool observations come from `KubeSimEnv.step`, so they are exactly what evaluation shows.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field

from tinyops.scenarios.base import Scenario
from tinyops.sim.env import KubeSimEnv
from tinyops.sim.tools import TOOL_NAMES, ToolBox, ToolCall, make_tool_callables
from tinyops.utils.log import get_logger

log = get_logger(__name__)

NO_INCIDENT = "Error: no active incident for this rollout"
ALREADY_SUBMITTED = "Diagnosis already recorded."


@dataclass
class ActiveRollout:
    scenario: Scenario
    env: KubeSimEnv = field(init=False)

    def __post_init__(self) -> None:
        # a large step budget: AgentTune's own max_steps_per_turn bounds the rollout
        self.env = KubeSimEnv(self.scenario.state, max_steps=10_000, alert_text=self.scenario.alert_text)

    def call(self, name: str, arguments: dict) -> str:
        if self.env.done:
            return ALREADY_SUBMITTED
        obs, _, _ = self.env.step(ToolCall(name=name, arguments=arguments))
        return obs


_ACTIVE: ContextVar[ActiveRollout | None] = ContextVar("tinyops_active_rollout", default=None)
_LATEST: ActiveRollout | None = None


def current() -> ActiveRollout | None:
    return _ACTIVE.get() or _LATEST


def bind(sc: Scenario | None) -> ActiveRollout | None:
    global _LATEST
    active = ActiveRollout(sc) if sc is not None else None
    _ACTIVE.set(active)
    _LATEST = active
    return active


class ScenarioRegistry:
    """alert text -> scenario. Alert texts must be unique (they are the rollout's identity)."""

    def __init__(self, scenarios: list[Scenario]) -> None:
        self.by_alert: dict[str, Scenario] = {}
        for sc in scenarios:
            if sc.alert_text in self.by_alert:
                raise ValueError(f"duplicate alert text, cannot identify rollouts: {sc.alert_text!r}")
            self.by_alert[sc.alert_text] = sc

    def lookup(self, conversation: list[dict]) -> Scenario | None:
        for m in conversation:
            if m.get("role") == "user":
                return self.by_alert.get(str(m.get("content", "")).strip())
        return None


def make_pre_step_hook(registry: ScenarioRegistry, history: list[str] | None = None) -> Callable:
    """`pre_step_hook(iteration, conversation, tools)`; binds the scenario at iteration 0."""

    def hook(iteration: int, conversation: list[dict], tools: object = None) -> None:
        if iteration != 0:
            return
        sc = registry.lookup(conversation)
        if sc is None:
            log.warning("rollout alert not in the registry: tools will report no active incident")
        bind(sc)
        if history is not None:
            history.append(sc.id if sc else "<unbound>")

    return hook


def make_rollout_tools() -> list[Callable]:
    """The 10 KubeSim tools as plain functions with typed signatures and Google docstrings.

    They accept any keyword arguments (`__signature__` keeps AgentTune's schema generation
    identical) and answer through the bound environment, so a wrong argument name produces the
    same error text as in evaluation instead of a Python TypeError.
    """
    templates = {f.__name__: f for f in make_tool_callables(ToolBox.__new__(ToolBox))}
    assert list(templates) == TOOL_NAMES
    tools: list[Callable] = []
    for name, tpl in templates.items():
        sig = inspect.signature(tpl)
        order = list(sig.parameters)

        def tool(*args, _name: str = name, _order: list[str] = order, **kwargs) -> str:
            active = current()
            if active is None:
                return NO_INCIDENT
            arguments = {**dict(zip(_order, args, strict=False)), **kwargs}
            return active.call(_name, arguments)

        tool.__name__ = name
        tool.__qualname__ = name
        tool.__doc__ = tpl.__doc__
        tool.__annotations__ = dict(tpl.__annotations__)
        tool.__signature__ = sig  # type: ignore[attr-defined]
        tools.append(tool)
    return tools
