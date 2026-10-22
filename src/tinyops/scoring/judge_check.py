"""Optional LLM-judge cross-check of the keyword matcher (validation split ONLY, off by default).

Estimates how often the deterministic matcher agrees with an LLM judge; the rate is reported in
docs/METRICS.md. It is never used for any test result.
"""

from __future__ import annotations

from collections.abc import Callable

from tinyops.scenarios.base import Scenario
from tinyops.scoring.diagnosis import match_diagnosis
from tinyops.sim.env import Episode

JUDGE_PROMPT = """You grade an SRE assistant's diagnosis of a Kubernetes incident.

Hidden ground truth:
- failure type: {failure_type}
- root cause object: {obj}
- key evidence: {evidence}

The assistant's diagnosis:
{root_cause}

Does the diagnosis correctly name the true root cause (it may use different wording)?
Answer with exactly one word: YES or NO."""


def litellm_complete(model: str = "gemini/gemini-2.5-flash") -> Callable[[str], str]:
    """A `prompt -> text` function backed by litellm (needs the provider's API key)."""
    import litellm

    def complete(prompt: str) -> str:
        r = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}],
                               temperature=0, max_tokens=5)
        return r.choices[0].message.content or ""

    return complete


def judge_agreement(
    pairs: list[tuple[Scenario, Episode]], complete: Callable[[str], str]
) -> dict:
    """Agreement between matcher success and the judge on submitted diagnoses (val only)."""
    if any(sc.split != "val" for sc, _ in pairs):
        raise ValueError("the LLM-judge cross-check may only run on the val split")
    agree = total = 0
    for sc, ep in pairs:
        if not ep.diagnosis:
            continue
        gt = sc.state.ground_truth
        verdict = complete(JUDGE_PROMPT.format(
            failure_type=gt.failure_type, obj=gt.root_cause_object,
            evidence="; ".join(gt.key_evidence), root_cause=ep.diagnosis.get("root_cause", "")))
        judge_yes = verdict.strip().upper().startswith("YES")
        # the judge sees only the root cause, so compare it with the root-cause part of the matcher
        m = match_diagnosis(ep.diagnosis, gt, ep.tool_outputs_seen)
        total += 1
        agree += int(judge_yes == (m.keywords_ok and m.type_confusion is None))
    return {"n": total, "agreement": agree / total if total else None}
