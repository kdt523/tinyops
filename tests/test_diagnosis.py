import random

import pytest

from tests.fixtures.states import make_state
from tinyops.scenarios.failure_types import FAILURE_TYPE_NAMES, FAILURE_TYPES
from tinyops.scenarios.names import TRAIN_POOLS
from tinyops.scoring.diagnosis import GENERIC_KEYWORDS, match_diagnosis, normalize

GT = make_state().ground_truth  # crashloop_missing_env, var DATABASE_URL
OBS = ["NAME  READY\ncheckout-api-7d9f4c-x2k9z  0/1", "Traceback ...\nKeyError: 'DATABASE_URL'"]


def diag(root, evidence=("KeyError: 'DATABASE_URL'",), fix="set it"):
    return {"root_cause": root, "evidence": list(evidence), "recommended_fix": fix}


def test_correct_phrasing_succeeds():
    r = match_diagnosis(
        diag("checkout-api crashes because the DATABASE_URL environment variable is not set."),
        GT, OBS)
    assert (r.submitted, r.keywords_ok, r.type_confusion, r.grounding, r.success) == (
        True, True, None, 1.0, True)


def test_missing_keyword_group_fails():
    r = match_diagnosis(diag("The environment variable is not set."), GT, OBS)
    assert not r.keywords_ok and not r.success
    r = match_diagnosis(diag("DATABASE_URL looks wrong."), GT, OBS)
    assert not r.keywords_ok and not r.success


def test_other_types_keywords_flag_type_confusion():
    r = match_diagnosis(
        diag("The service selector does not match the pod labels so there are no endpoints."),
        GT, OBS)
    assert not r.keywords_ok and r.type_confusion == "service_selector_mismatch"
    assert not r.success


def test_list_every_cause_fails():
    root = ("It could be the DATABASE_URL environment variable, or the service selector label "
            "with empty endpoints, or an OOM, or a hung deadlock.")
    r = match_diagnosis(diag(root), GT, OBS)
    assert r.keywords_ok and r.type_confusion is not None and not r.success


def test_incidental_mention_of_another_type_does_not_confuse():
    root = ("Pods are stuck in CrashLoopBackOff because the DATABASE_URL environment variable "
            "is not set.")
    r = match_diagnosis(diag(root), GT, OBS)
    assert r.type_confusion is None and r.success  # 'stuck' alone is weaker than the real match


def test_ungrounded_evidence_fails():
    r = match_diagnosis(
        diag("DATABASE_URL environment variable missing", evidence=["I made this line up"]),
        GT, OBS)
    assert r.keywords_ok and r.grounding == 0.0 and not r.success


def test_grounding_is_fraction_and_threshold_is_half():
    root = "DATABASE_URL environment variable missing"
    half = match_diagnosis(diag(root, evidence=["KeyError: 'DATABASE_URL'", "invented"]), GT, OBS)
    assert half.grounding == 0.5 and half.success
    third = match_diagnosis(diag(root, evidence=["KeyError: 'DATABASE_URL'", "x", "y"]), GT, OBS)
    assert third.grounding == pytest.approx(1 / 3) and not third.success


def test_tiny_evidence_strings_cannot_be_grounded():
    """'e' is a substring of every observation; it must not count (reward-hacking guard)."""
    root = "DATABASE_URL environment variable missing"
    for junk in ("e", "the", "ok", "  x  ", "a b c"):
        r = match_diagnosis(diag(root, evidence=[junk]), GT, ["the pods are listed here"])
        assert r.grounding == 0.0 and not r.success, junk
    ok = match_diagnosis(diag(root, evidence=["DATABASE_URL"]), GT, OBS)
    assert ok.grounding == 1.0


def test_empty_evidence_and_no_submission():
    root = "DATABASE_URL environment variable missing"
    r = match_diagnosis(diag(root, evidence=[]), GT, OBS)
    assert r.submitted and r.grounding == 0.0 and not r.success
    for none in (None, {}):
        r = match_diagnosis(none, GT, OBS)
        assert (r.submitted, r.keywords_ok, r.success, r.grounding) == (False, False, False, 0.0)


def test_normalization_handles_case_and_punctuation():
    assert normalize("  KeyError:  'DATABASE_URL'!! ") == "keyerror database_url"
    assert normalize("a-b_c, D.") == "a-b_c d"
    r = match_diagnosis(
        diag("ENVIRONMENT VARIABLE: database_url -- is MISSING!!!",
             evidence=["keyerror:    'database_url'"]), GT, OBS)
    assert r.success


def test_stem_matching_accepts_inflections():
    gt = FAILURE_TYPES["node_memory_pressure_eviction"](random.Random(1), "easy", TRAIN_POOLS).ground_truth
    obs = ["The node was low on resource: memory"]
    r = match_diagnosis(diag("Pods were evicted from the node", evidence=[obs[0]]), gt, obs)
    assert r.keywords_ok and r.success
    # but a word that merely contains the term inside is not a match
    gt2 = FAILURE_TYPES["oom_killed"](random.Random(1), "easy", TRAIN_POOLS).ground_truth
    assert not match_diagnosis(diag("the room is full", evidence=["OOMKilled"]), gt2,
                               ["OOMKilled"]).keywords_ok
    assert match_diagnosis(diag("the pod was OOMKilled", evidence=["OOMKilled"]), gt2,
                           ["OOMKilled"]).success


@pytest.mark.parametrize("ft", FAILURE_TYPE_NAMES)
def test_generic_keywords_match_generators(ft):
    gt = FAILURE_TYPES[ft](random.Random(2), "medium", TRAIN_POOLS).ground_truth
    generic = GENERIC_KEYWORDS[ft]
    assert gt.root_cause_keywords[: len(generic)] == generic
    assert len(gt.root_cause_keywords) - len(generic) in (0, 1)
    if ft in ("crashloop_missing_env", "bad_configmap_value"):
        assert len(gt.root_cause_keywords) == len(generic) + 1
