"""Policy interface: anything that maps a conversation to the next model text."""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel

from tinyops.sim.tools import ToolCall


class PolicyOutput(BaseModel):
    text: str
    tool_call: ToolCall | None = None
    usage: dict[str, Any] = {}


class Policy(Protocol):
    name: str

    def act(self, messages: list[dict]) -> PolicyOutput: ...


def format_tool_call_text(call: ToolCall) -> str:
    """Tool call as model text in AgentTune's Tiny Aya fallback list format."""
    return json.dumps(
        [{"tool_name": call.name, "parameters": call.arguments}], ensure_ascii=False
    )


class ScriptedSequencePolicy:
    """Replays a fixed list of calls (tests). Emits prose once the list is exhausted."""

    def __init__(self, calls: list[ToolCall], name: str = "scripted_sequence") -> None:
        self.name = name
        self._calls = list(calls)
        self._i = 0

    def act(self, messages: list[dict]) -> PolicyOutput:
        if self._i >= len(self._calls):
            return PolicyOutput(text="I am done investigating.")
        call = self._calls[self._i]
        self._i += 1
        return PolicyOutput(text=format_tool_call_text(call), tool_call=call)
