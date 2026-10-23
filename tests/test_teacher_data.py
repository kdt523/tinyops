import json

import pytest

from tests.test_run_eval import StubPolicy
from tinyops.eval.run_eval import run_batched
from tinyops.policies.base import PolicyOutput
from tinyops.scenarios.build_dataset import build_split, load_scenarios
from tinyops.sim.env import Episode
from tinyops.train.teacher_data import (
    build_extra_train,
    canonicalize,
    filter_teacher_episodes,
    teacher_rows,
)
from tinyops.utils.io import read_jsonl


class Teacher(StubPolicy):
    """Oracle replay with sloppy JSON spacing; one scenario gets a prose answer first."""

    name = "teacher"

    def act(self, messages):
        out = super().act(messages)
        text = json.dumps(json.loads(out.text), separators=(",", ":"))
        return PolicyOutput(text=text, tool_call=out.tool_call, usage=out.usage)


class Fenced(StubPolicy):
    name = "fenced"

    def act(self, messages):
        out = super().act(messages)
        return PolicyOutput(text=f"```json\n{out.text}\n```", tool_call=out.tool_call)


@pytest.fixture
def scs(built_data):
    # the replaying stub is an oracle: its "ignored an instruction" note is (by design) ungrounded
    return [s for s in load_scenarios("val", built_data) if not s.injected][:6]


def run(policy_cls, scs, tmp_path):
    out = tmp_path / "t.jsonl"
    run_batched(scs, policy_cls(scs), out, batch_size=3)
    return [Episode.model_validate(r) for r in read_jsonl(out)]


def test_clean_teacher_episodes_are_kept_and_canonicalised(scs, tmp_path):
    eps = run(Teacher, scs, tmp_path)
    kept, stats = filter_teacher_episodes(eps, {s.id: s for s in scs})
    assert stats == {"total": 6, "kept": 6, "rejected": {}}
    assert all(not s.raw_model_text.startswith("[{\"tool_name\":\"") for e in kept for s in e.steps)
    assert kept[0].steps[0].raw_model_text.startswith('[{"tool_name": ')  # canonical spacing
    assert eps[0].steps[0].raw_model_text.startswith('[{"tool_name":"')  # original untouched


def test_fenced_or_failed_episodes_are_rejected(scs, tmp_path):
    kept, stats = filter_teacher_episodes(run(Fenced, scs, tmp_path), {s.id: s for s in scs})
    assert kept == [] and stats["rejected"] == {"not_plain_fallback_format": 6}


def test_rejection_reasons(scs, tmp_path):
    eps = run(Teacher, scs, tmp_path)
    by_id = {s.id: s for s in scs}
    bad = eps[0].model_copy(deep=True)
    bad.diagnosis = {"root_cause": "pods are unhealthy", "evidence": ["x"], "recommended_fix": "-"}
    unsafe = eps[1].model_copy(deep=True)
    unsafe.steps[0].is_unsafe = True
    dup = eps[2].model_copy(deep=True)
    dup.steps[1].is_duplicate = True
    ungrounded = eps[3].model_copy(deep=True)
    ungrounded.diagnosis = {**ungrounded.diagnosis, "evidence": [*ungrounded.diagnosis["evidence"], "made up evidence line"]}
    _, stats = filter_teacher_episodes([bad, unsafe, dup, ungrounded], by_id)
    assert stats["rejected"] == {"not_successful": 1, "unsafe": 1, "redundant_calls": 1, "ungrounded": 1}


def test_canonicalize_is_idempotent(scs, tmp_path):
    ep = run(Teacher, scs, tmp_path)[0]
    assert canonicalize(canonicalize(ep)).model_dump() == canonicalize(ep).model_dump()


def test_extra_train_scenarios_never_collide_with_other_splits(built_data, tmp_path):
    extra = build_extra_train(5678, tmp_path)
    assert len(extra) == 1215 and all(s.id.startswith("train_extra-") and s.split == "train" for s in extra)
    others = {s.id for sp in ("train", "val", "test", "test_injection", "test_ood")
              for s in load_scenarios(sp, built_data)}
    assert not {s.id for s in extra} & others
    states = {s.state.model_dump_json() for sp in ("train", "val", "test", "test_injection", "test_ood")
              for s in load_scenarios(sp, built_data)}
    assert not {s.state.model_dump_json() for s in extra[:200]} & states
    assert (tmp_path / "train_extra" / "scenarios.jsonl").exists()
    assert next(s.id for s in build_split("train", 1234)).startswith("train-")


def test_teacher_rows_are_flagged_and_use_the_training_prompt(tmp_path):
    from tinyops.policies.parsing import parse_tool_call
    from tinyops.policies.prompt import build_system_prompt

    extra = build_extra_train(5678, tmp_path)[:4]
    eps = run(Teacher, extra, tmp_path)
    by_id = {s.id: s for s in extra}
    kept, stats = filter_teacher_episodes(eps, by_id)
    assert stats["kept"] == 4
    rows = teacher_rows(kept, by_id)
    assert {r["source"] for r in rows} == {"teacher"}
    assert all(r["id"].startswith("train_extra-") and r["split"] == "train" for r in rows)
    for r in rows:
        assert r["messages"][0]["content"] == build_system_prompt()
        calls = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
        assert all(parse_tool_call(c).ok and c.startswith('[{"tool_name": ') for c in calls)
