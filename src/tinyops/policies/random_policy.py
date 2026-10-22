"""Random baseline for sanity checks only: plausible-looking calls, no investigation."""

from __future__ import annotations

import random
import re

from tinyops.policies.base import PolicyOutput, format_tool_call_text
from tinyops.scenarios.names import OOD_POOLS, TRAIN_POOLS
from tinyops.sim.tools import READ_TOOLS, WRITE_TOOLS, ToolCall

_POD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*-[0-9a-f]{6}-[a-z0-9]{5}")
_ALL_NS = (*TRAIN_POOLS.namespaces, *OOD_POOLS.namespaces)
_CAUSES = (
    "The service is unhealthy and needs investigation.",
    "Something is wrong with the pods in this namespace.",
    "The deployment is misconfigured.",
    "The application is failing under load.",
)


class RandomPolicy:
    name = "random"

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.submit_after = rng.randint(1, 5)

    def _namespace(self, messages: list[dict]) -> str:
        alert = messages[1]["content"] if len(messages) > 1 else ""
        found = [ns for ns in _ALL_NS if re.search(rf"\b{re.escape(ns)}\b", alert)]
        return self.rng.choice(found or list(_ALL_NS))

    def act(self, messages: list[dict]) -> PolicyOutput:
        rng = self.rng
        turns = sum(1 for m in messages if m["role"] == "assistant")
        seen = "\n".join(m["content"] for m in messages[2:] if m["role"] == "user")
        pods = sorted(set(_POD_RE.findall(seen)))
        ns = self._namespace(messages)
        pod = rng.choice(pods) if pods else "unknown-pod"
        if turns >= self.submit_after:
            lines = [ln.strip() for ln in seen.split("\n") if ln.strip()]
            evidence = rng.sample(lines, min(len(lines), rng.randint(1, 2))) or ["no evidence"]
            call = ToolCall(name="submit_diagnosis", arguments={
                "root_cause": rng.choice(_CAUSES), "evidence": evidence,
                "recommended_fix": "Investigate further."})
        else:
            name = rng.choice([*READ_TOOLS] * 12 + sorted(WRITE_TOOLS))
            args = {
                "list_pods": {"namespace": ns},
                "describe_pod": {"namespace": ns, "pod": pod},
                "get_events": {"namespace": ns},
                "get_logs": {"namespace": ns, "pod": pod, "tail": rng.choice([10, 20, 50])},
                "get_metrics": {"namespace": ns, "pod": pod},
                "check_service": {"namespace": ns, "service": ns},
                "get_configmap": {"namespace": ns, "name": f"{ns}-config"},
                "restart_deployment": {"namespace": ns, "deployment": ns, "reason": "random"},
                "delete_namespace": {"namespace": ns},
            }[name]
            call = ToolCall(name=name, arguments=args)
        return PolicyOutput(text=format_tool_call_text(call), tool_call=call)
