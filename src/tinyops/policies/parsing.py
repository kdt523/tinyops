"""Strict tool-call parser for model text.

Accepted formats: AgentTune's fallback list `[{"tool_name": ..., "parameters": {...}}]`, Cohere
`<|START_ACTION|>[...]<|END_ACTION|>`, a single JSON object, and any of them in a markdown fence
(`*_fenced`). Malformed JSON is never repaired: it is a parse failure.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

from tinyops.sim.tools import ToolCall

NAME_KEYS = ("tool_name", "name", "tool")
PARAM_KEYS = ("parameters", "arguments", "args")
_FENCE = re.compile(r"```[a-zA-Z0-9_-]*[ \t]*\n?(.*?)```", re.DOTALL)
_ACTION = re.compile(r"<\|START_ACTION\|>(.*?)(?:<\|END_ACTION\|>|$)", re.DOTALL)
_LOOKS_LIKE_CALL = re.compile(r'"(?:tool_name|name|tool)"\s*:')
_OPEN, _CLOSE = "[{", "]}"


class ParseResult(BaseModel):
    ok: bool
    call: ToolCall | None = None
    format: str | None = None
    error: str | None = None
    error_kind: str | None = None  # empty_output | no_tool_call | malformed_json | invalid_structure
    n_calls_found: int = 0


def balanced_end(text: str, start: int) -> int | None:
    """Index just past the bracket structure opened at `start`, or None if unterminated.

    Respects JSON strings and escapes; counts only the matching bracket kinds.
    """
    stack: list[str] = []
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in _OPEN:
            stack.append(_CLOSE[_OPEN.index(ch)])
        elif ch in _CLOSE:
            if not stack or stack.pop() != ch:
                return None
            if not stack:
                return i + 1
    return None


def _first(d: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _as_call(item) -> tuple[ToolCall | None, str | None]:
    """(call, error) for one JSON element."""
    if not isinstance(item, dict):
        return None, "call is not a JSON object"
    name = _first(item, NAME_KEYS)
    if not isinstance(name, str) or not name.strip():
        return None, "missing tool name (tool_name/name/tool)"
    params = _first(item, PARAM_KEYS)
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return None, f"parameters must be a JSON object, got {type(params).__name__}"
    return ToolCall(name=name.strip(), arguments=params), None


def _is_call_shaped(value) -> bool:
    items = value if isinstance(value, list) else [value]
    return bool(items) and all(isinstance(i, dict) and _first(i, NAME_KEYS) is not None for i in items)


def _scan(text: str):
    """First tool-call-shaped JSON value in `text`: (value, error_kind, error, end_index)."""
    i = 0
    while i < len(text):
        if text[i] not in _OPEN:
            i += 1
            continue
        end = balanced_end(text, i)
        span = text[i:end] if end else text[i:]
        try:
            if end is None:
                raise ValueError("unterminated")
            value = json.loads(span)
        except ValueError:
            if _LOOKS_LIKE_CALL.search(span):
                kind = "unterminated JSON" if end is None else "invalid JSON"
                return None, "malformed_json", f"{kind} in tool call", len(text)
            i += 1 if end is None else len(span)
            continue
        if isinstance(value, (list, dict)) and _is_call_shaped(value):
            return value, None, None, i + len(span)
        i += len(span)  # valid JSON but not a call (e.g. [1, 2], prose brackets): skip it
    return None, None, None, len(text)


def _parse_segment(text: str, suffix: str = "") -> ParseResult | None:
    """Parse one text segment; None if it contains no tool-call-looking JSON at all."""
    value, kind, err, end = _scan(text)
    if kind:
        return ParseResult(ok=False, error=err, error_kind=kind)
    if value is None:
        return None
    items = value if isinstance(value, list) else [value]
    calls = []
    for item in items:
        call, why = _as_call(item)
        if call is None:
            return ParseResult(ok=False, error=why, error_kind="invalid_structure",
                               n_calls_found=len(items))
        calls.append(call)
    extra, rest = 0, text[end:]
    while rest:  # further call-shaped JSON values later in the text (executed one per turn)
        more, bad, _, end2 = _scan(rest)
        if more is None or bad:
            break
        extra += len(more) if isinstance(more, list) else 1
        rest = rest[end2:]
    fmt = "fallback_list" if isinstance(value, list) else "json_object"
    return ParseResult(ok=True, call=calls[0], format=fmt + suffix,
                       n_calls_found=len(calls) + extra)


def parse_tool_call(text: str) -> ParseResult:
    if text is None or not text.strip():
        return ParseResult(ok=False, error="empty model output", error_kind="empty_output")
    m = _ACTION.search(text)
    if m:
        res = _parse_segment(m.group(1))
        if res is not None:
            if res.ok:
                res.format = "cohere_action"
            return res
    for fence in _FENCE.finditer(text):
        res = _parse_segment(fence.group(1), "_fenced")
        if res is not None:
            return res
    res = _parse_segment(text)
    if res is not None:
        return res
    return ParseResult(ok=False, error="no tool call found in model output", error_kind="no_tool_call")


def has_complete_call_json(text: str) -> bool:
    """True once `text` holds a finished, call-looking JSON structure (generation can stop).

    Unterminated structures return False (keep generating); balanced structures that do not
    look like a tool call (prose brackets) are skipped.
    """
    i = 0
    while i < len(text):
        if text[i] not in _OPEN:
            i += 1
            continue
        end = balanced_end(text, i)
        if end is None:
            return False
        if _LOOKS_LIKE_CALL.search(text[i:end]):
            return True
        i = end
    return False
