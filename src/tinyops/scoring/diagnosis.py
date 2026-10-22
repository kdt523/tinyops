"""Deterministic diagnosis matcher: keywords, type confusion, grounding."""

from __future__ import annotations

import re

from pydantic import BaseModel

from tinyops.sim.cluster import GroundTruth

# Keyword groups (AND of ORs) of each failure type without the scenario-specific group
# (the env var name / config key). Used to detect that an answer describes a different failure.
GENERIC_KEYWORDS: dict[str, list[list[str]]] = {
    "crashloop_missing_env": [["env", "environment variable", "variable"]],
    "oom_killed": [["oom", "out of memory", "memory limit"]],
    "image_pull_backoff": [["image", "tag"], ["pull", "not found", "does not exist"]],
    "readiness_probe_misconfig": [["readiness", "probe"], ["port", "path"]],
    "service_selector_mismatch": [["selector", "label"], ["service", "endpoint"]],
    "pending_insufficient_memory": [["insufficient", "not enough"], ["memory", "resources"]],
    "node_memory_pressure_eviction": [["node"], ["memory pressure", "low on resource", "evict"]],
    "bad_configmap_value": [["configmap", "config"]],
    "hung_process_restart": [["deadlock", "hung", "stuck"]],
}


MIN_EVIDENCE_CHARS = 6


class DiagnosisResult(BaseModel):
    submitted: bool
    keywords_ok: bool
    type_confusion: str | None
    grounding: float
    success: bool


def normalize(text: str) -> str:
    """Lowercase, punctuation (except `_` and `-`) -> space, collapse whitespace."""
    text = re.sub(r"[^\w\s-]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def _present(norm_text: str, term: str) -> bool:
    """Term occurs at a word start (so `evict` matches `evicted`, `oom` matches `oomkilled`)."""
    return re.search(rf"(?<!\w){re.escape(normalize(term))}", norm_text) is not None


def keyword_score(text: str, groups: list[list[str]]) -> tuple[bool, int]:
    """(every group has >=1 term present, number of distinct terms present)."""
    norm = normalize(text)
    present = [[t for t in g if _present(norm, t)] for g in groups]
    ok = all(present)
    return ok, len({t for g in present for t in g})


def match_diagnosis(
    diagnosis: dict | None, gt: GroundTruth, observations: list[str]
) -> DiagnosisResult:
    if not diagnosis:
        return DiagnosisResult(submitted=False, keywords_ok=False, type_confusion=None,
                               grounding=0.0, success=False)
    root = str(diagnosis.get("root_cause", ""))
    keywords_ok, correct_terms = keyword_score(root, gt.root_cause_keywords)

    # does the answer fit another failure type at least as well? (blocks "list every cause")
    confusion: str | None = None
    best_other = -1
    for other, groups in GENERIC_KEYWORDS.items():
        if other == gt.failure_type:
            continue
        ok, n = keyword_score(root, groups)
        if ok and n > best_other and (not keywords_ok or n >= correct_terms):
            confusion, best_other = other, n

    evidence = diagnosis.get("evidence") or []
    if evidence:
        obs_norm = [normalize(o) for o in observations]
        # items shorter than MIN_EVIDENCE_CHARS ("e", "the") are substrings of everything:
        # they can never count as grounded (anti reward-hacking)
        hits = sum(
            1 for e in evidence
            if len(n := normalize(e)) >= MIN_EVIDENCE_CHARS and any(n in o for o in obs_norm)
        )
        grounding = hits / len(evidence)
    else:
        grounding = 0.0
    success = keywords_ok and confusion is None and grounding >= 0.5
    return DiagnosisResult(submitted=True, keywords_ok=keywords_ok, type_confusion=confusion,
                           grounding=grounding, success=success)
