"""Training rewards for GRPO. Each rollout is replayed through the strict KubeSim environment.

A reward is a pure function of (the rollout's assistant turns, the scenario). The turns come from
AgentTune's rollout output: `trajectories` (step.thought = raw model text per turn, plus
final_response) or, failing that, `conversations`. The scenario comes from the dataset column
`scenario_id`. No state is shared with the tools, so rewards stay consistent with evaluation.

Signature (verified against AgentTune/TRL, docs/api_notes.md): `fn(completions, **kwargs)` where
kwargs holds `prompts`, every dataset column, and the rollout extras as same-length lists.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinyops.policies.base import PolicyOutput
from tinyops.policies.parsing import parse_tool_call
from tinyops.scenarios.base import Scenario
from tinyops.scoring.diagnosis import match_diagnosis
from tinyops.sim.env import Episode, EpisodeSession
from tinyops.utils.io import append_jsonl

DEFAULT_WEIGHTS = {"diagnosis": 1.0, "format": 0.5, "schema": 0.5, "efficiency": 0.3,
                   "grounding": 0.3, "safety": 1.0}
NAMES = tuple(DEFAULT_WEIGHTS)
PROSE_KINDS = {"no_tool_call", "empty_output"}


def turn_texts(kwargs: dict, i: int) -> list[str]:
    """Raw assistant turn texts of rollout i, in order."""
    trajs = kwargs.get("trajectories")
    if trajs:
        t = trajs[i]
        texts = [s.thought or "" for s in t.steps if getattr(s, "action", None)]
        final = (t.final_response or "").strip()
        if final:
            texts.append(final)
        return texts
    convs = kwargs.get("conversations")
    if convs:
        return [str(m.get("content") or "") for m in convs[i] if m.get("role") == "assistant"]
    raise KeyError("reward kwargs carry neither `trajectories` nor `conversations`")


@dataclass
class Replay:
    episode: Episode
    post_submit_calls: int


def replay(texts: list[str], sc: Scenario, max_steps: int = 8) -> Replay:
    """Run the assistant turns through KubeSim with the strict parser."""
    session = EpisodeSession(sc.state, sc.alert_text, "rollout", max_steps, 0, sc.id)
    post = 0
    for text in texts:
        res = parse_tool_call(text)
        if session.env.done:
            post += int(res.ok and session.env.termination == "submitted")
            continue
        session.apply(PolicyOutput(text=text, tool_call=res.call))
    # a rollout that stopped without submitting or hitting a limit is reported as "policy_error"
    return Replay(session.episode(), post)


def components(rp: Replay, sc: Scenario) -> dict[str, float]:
    ep, gt = rp.episode, sc.state.ground_truth
    diag = match_diagnosis(ep.diagnosis, gt, ep.tool_outputs_seen)
    steps = ep.steps
    # diagnosis: 1.0 correct and grounded; 0.3 right words but ungrounded; else 0
    if diag.success:
        r_diag = 1.0
    elif diag.keywords_ok and diag.type_confusion is None and diag.grounding < 0.5:
        r_diag = 0.3
    else:
        r_diag = 0.0
    # format: +0.2 when every turn parsed; unparseable turns cost 0.2 (broken call) or 0.3 (prose)
    bad = [s for s in steps if not s.parsed_ok]
    if not steps:
        r_fmt = -0.3
    elif not bad:
        r_fmt = 0.2
    else:
        pen = sum(0.3 if parse_tool_call(s.raw_model_text or "").error_kind in PROSE_KINDS else 0.2
                  for s in bad)
        r_fmt = -min(pen, 0.6)
    # schema: -0.2 per invalid call, -0.3 per hallucinated tool, floor -1.0
    r_schema = -min(1.0, sum(0.3 if not s.tool_exists else 0.2
                             for s in steps if s.parsed_ok and not s.schema_valid))
    # efficiency
    n_calls = sum(1 for s in steps if s.parsed_ok and s.tool_exists)
    r_eff = -0.1 * sum(s.is_duplicate for s in steps) - 0.1 * rp.post_submit_calls
    if ep.termination == "max_steps":
        r_eff -= 0.3
    if ep.termination == "submitted" and n_calls <= len(gt.relevant_tools) + 1:
        r_eff += 0.1
    r_eff = max(r_eff, -1.0)
    # grounding
    r_ground = 0.3 * diag.grounding if diag.submitted else 0.0
    # safety
    r_safe = (-2.0 if any(s.is_unsafe for s in steps) else 0.0) \
        - 0.5 * sum(s.is_unnecessary_write for s in steps) \
        + (0.2 if any(s.is_correct_fix for s in steps) else 0.0)
    raw = {"diagnosis": r_diag, "format": r_fmt, "schema": r_schema, "efficiency": r_eff,
           "grounding": r_ground, "safety": r_safe}
    return {k: float(v) for k, v in raw.items()}  # TRL needs real floats, never ints


def total(comp: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights[k] * comp[k] for k in NAMES)


class RewardSet:
    """The six reward components, their weighted sum, and an optional breakdown log."""

    def __init__(self, scenarios: dict[str, Scenario], weights: dict[str, float] | None = None,
                 log_path: Path | None = None, max_steps: int = 8) -> None:
        self.scenarios = scenarios
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.log_path, self.max_steps = log_path, max_steps
        self._cache: dict[tuple, dict[str, float]] = {}

    def breakdown(self, kwargs: dict) -> list[dict]:
        ids = kwargs["scenario_id"]
        out = []
        for i, sid in enumerate(ids):
            texts = turn_texts(kwargs, i)
            key = (sid, tuple(texts))
            if key not in self._cache:
                if len(self._cache) > 8192:
                    self._cache.clear()
                self._cache[key] = components(replay(texts, self.scenarios[sid], self.max_steps),
                                              self.scenarios[sid])
            comp = self._cache[key]
            out.append({"scenario_id": sid, "components": comp,
                        "total": total(comp, self.weights), "texts": texts})
        return out

    def component(self, name: str) -> Callable[..., list[float]]:
        def fn(completions: Any = None, **kwargs: Any) -> list[float]:
            return [b["components"][name] for b in self.breakdown(kwargs)]

        fn.__name__ = f"r_{name}"
        return fn

    def combined(self, completions: Any = None, **kwargs: Any) -> list[float]:
        rows = self.breakdown(kwargs)
        if self.log_path is not None:
            for r in rows:
                append_jsonl(self.log_path, {"scenario_id": r["scenario_id"], "total": r["total"],
                                             **r["components"], "texts": r["texts"]})
        log_metric = kwargs.get("log_metric")
        if callable(log_metric) and rows:
            for name in NAMES:
                try:
                    log_metric(f"reward/{name}", sum(r["components"][name] for r in rows) / len(rows))
                except TypeError:
                    break
        return [r["total"] for r in rows]

    def reward_funcs(self) -> list[Callable[..., list[float]]]:
        """[combined]: one callable AgentTune passes through unchanged (components are logged inside)."""
        return [self.combined]
