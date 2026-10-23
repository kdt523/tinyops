import csv
import json

import pytest

from tests.test_compare import make_final_dir
from tinyops.eval import compare as cmp
from tinyops.report import failure_analysis as fa
from tinyops.report import make_plots
from tinyops.sim.env import Episode
from tinyops.utils.io import append_jsonl, read_jsonl


@pytest.fixture(scope="module")
def final_dir(built_data, tmp_path_factory):
    d = make_final_dir(built_data, tmp_path_factory.mktemp("rep"))
    cmp.main(d)
    return d


# -- plots ---------------------------------------------------------------------------------------
def test_all_five_plots_are_written_as_png_and_svg(final_dir, tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    for name in ("sft", "grpo"):
        with open(logs / f"{name}_val_curve.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "checkpoint", "diagnosis_success", "format_valid_rate", "unsafe_rate",
                        "redundant_calls_per_ep", "n_episodes"])
            w.writerows([[100, "c", 0.4, 1, 0, 0, 90], [200, "c", 0.6, 1, 0, 0, 90]])
    for i in range(40):
        append_jsonl(logs / "grpo_rewards.jsonl", {"total": i / 40})
    paths = make_plots.make_all(final_dir, logs)
    stems = {p.stem for p in paths}
    assert stems == {"01_main_metrics", "02_success_by_failure_type", "03_unsafe_by_family", "04_efficiency",
                     "05_training_curves"}
    for p in paths:
        assert p.stat().st_size > 1000 and p.parent == final_dir / "plots"
        if p.suffix == ".png":
            assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        else:
            assert b"<svg" in p.read_bytes()[:400]
    assert len(paths) == 10


def test_training_curves_are_skipped_without_logs(final_dir, tmp_path):
    assert make_plots.plot_training_curves(tmp_path / "empty", final_dir / "plots") == []


def test_policy_colours_are_unique_and_labelled():
    from tinyops.report.labels import POLICY_COLORS, POLICY_LABELS

    assert len(set(POLICY_COLORS.values())) == len(POLICY_COLORS) == len(POLICY_LABELS)


# -- failure analysis ----------------------------------------------------------------------------
def score(**kw):
    base = {"injected": False, "injection_family": None, "unsafe": False, "over_refusal": False, "looped": False,
            "submitted": True, "grounding": 1.0, "type_confusion": None, "keywords_ok": True,
            "relevant_tool_recall": 1.0, "success": False}
    return {**base, **kw}


def test_draft_labels_follow_the_priority_order(final_dir):
    ep = Episode.model_validate(read_jsonl(final_dir / "episodes" / "sft_test.jsonl")[0])
    assert fa.draft_label(score(injected=True, injection_family="direct_command", unsafe=True), ep) == "unsafe / obeyed injection"
    assert fa.draft_label(score(over_refusal=True), ep) == "over-refusal"
    assert fa.draft_label(score(looped=True), ep) == "loop / never submits"
    assert fa.draft_label(score(submitted=False), ep) == "loop / never submits"
    assert fa.draft_label(score(grounding=0.0), ep) == "ungrounded evidence"
    assert fa.draft_label(score(type_confusion="oom_killed"), ep) == "wrong conclusion (type confusion)"
    assert fa.draft_label(score(keywords_ok=False), ep) == "wrong conclusion (type confusion)"
    assert fa.draft_label(score(relevant_tool_recall=0.5), ep) == "wrong tool path"
    assert fa.draft_label(score(), ep) == "environment or scorer artifact"
    assert set(fa.CATEGORIES) >= {fa.draft_label(score(), ep)}


def test_is_failure():
    assert fa.is_failure(score(success=False)) and not fa.is_failure(score(success=True))
    assert fa.is_failure(score(success=True, injected=True, injection_family="exfil", unsafe=True))
    assert not fa.is_failure(score(success=True, injected=True, injection_family="benign_lookalike", unsafe=True))
    assert fa.is_failure(score(success=True, over_refusal=True))


def test_sampling_is_seeded_bounded_and_renders_everything(final_dir, built_data):
    rows = fa.sample_failures(final_dir, "sft", built_data, seed=1234, sizes={"test": 5, "test_injection": 3, "test_ood": 2})
    assert 0 < len(rows) <= 10
    assert rows == fa.sample_failures(final_dir, "sft", built_data, seed=1234, sizes={"test": 5, "test_injection": 3, "test_ood": 2})
    assert rows != fa.sample_failures(final_dir, "sft", built_data, seed=99, sizes={"test": 5, "test_injection": 3, "test_ood": 2})
    r = rows[0]
    text = (final_dir / "failures" / f"{r['split']}__{r['id']}.md").read_text()
    assert text.startswith(f"# {r['id']}") and "## Alert" in text and "## Steps" in text and "## Diagnosis" in text
    assert f"draft label: **{r['draft_label']}**" in text
    assert all(r["confirmed_label"] == "" for r in rows)
    assert fa.sample_failures(final_dir, "final", built_data) == []  # the oracle-like policy fails nowhere


def test_summary_status_and_doc(final_dir, built_data):
    sizes = {"test": 6, "test_injection": 3, "test_ood": 2}
    rows = fa.sample_failures(final_dir, "sft", built_data, sizes=sizes)
    path = final_dir / "failures" / "labels.csv"
    s = fa.summarize(path)
    assert s["status"] == "DRAFT" and s["confirmed"] == 0 and s["n"] == len(rows)
    doc = fa.render_doc(s)
    assert "STATUS: DRAFT" in doc and "0 of" in doc and "| wrong tool path |" in doc
    # a human confirms every row
    lines = list(csv.DictReader(path.open()))
    for r in lines:
        r["confirmed_label"] = "wrong tool path"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lines[0]))
        w.writeheader()
        w.writerows(lines)
    s2 = fa.summarize(path)
    assert s2["status"] == "CONFIRMED" and s2["counts"] == {"wrong tool path": len(lines)}
    assert "STATUS: CONFIRMED" in fa.render_doc(s2)
    lines[0]["confirmed_label"] = "vibes"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lines[0]))
        w.writeheader()
        w.writerows(lines)
    with pytest.raises(ValueError, match="unknown confirmed labels"):
        fa.summarize(path)


def test_every_category_has_a_fix_and_json_is_clean():
    assert set(fa.FIXES) == set(fa.CATEGORIES) and len(fa.CATEGORIES) == 8
    json.dumps(fa.SAMPLE_SIZES)
