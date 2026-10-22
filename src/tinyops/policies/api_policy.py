"""Closed/teacher model policy through LiteLLM (e.g. Gemini)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from tinyops.policies.base import PolicyOutput
from tinyops.policies.parsing import parse_tool_call
from tinyops.utils.log import get_logger

log = get_logger(__name__)


class APIPolicy:
    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash",
        temperature: float = 0,
        max_tokens: int = 256,
        rpm: int = 60,
        retries: int = 3,
        name: str | None = None,
        extra: dict[str, Any] | None = None,
        completion_fn: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.rpm, self.retries, self._sleep = rpm, retries, sleep
        self.name = name or model.replace("/", "-")
        self.extra = dict(extra or {})
        self._completion = completion_fn
        self._lock = threading.Lock()
        self._next_ok = 0.0

    def config(self) -> dict:
        return {"policy": "api", "name": self.name, "model_id": self.model,
                "temperature": self.temperature, "max_tokens": self.max_tokens, "rpm": self.rpm,
                "retries": self.retries, "extra": self.extra}

    def _call(self, messages: list[dict]) -> Any:
        if self._completion is None:
            import litellm

            self._completion = litellm.completion
        return self._completion(model=self.model, messages=messages,
                                temperature=self.temperature, max_tokens=self.max_tokens,
                                **self.extra)

    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_ok - now
            self._next_ok = max(now, self._next_ok) + 60.0 / self.rpm
        if wait > 0:
            self._sleep(wait)

    def act(self, messages: list[dict]) -> PolicyOutput:
        last: Exception | None = None
        for attempt in range(self.retries):
            self._throttle()
            t0 = time.perf_counter()
            try:
                resp = self._call(messages)
            except Exception as e:  # noqa: BLE001 - provider errors vary; retry with backoff
                last = e
                log.warning("API call failed (%s), attempt %d/%d", e, attempt + 1, self.retries)
                if attempt < self.retries - 1:
                    self._sleep(2.0**attempt)
                continue
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            return PolicyOutput(
                text=text, tool_call=parse_tool_call(text).call,
                usage={
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "latency_s": time.perf_counter() - t0,
                })
        raise RuntimeError(f"API call failed after {self.retries} attempts: {last}")
