import json

import pytest

from tinyops.eval import compare as cmp
from tinyops.eval.run_eval import POLICIES, run_policy
from tinyops.eval.score_episodes import score_file
from tinyops.report.make_tables import cell, p_str, readme_table, render_tables
from tinyops.scenarios.build_dataset import load_scenarios


def half(sc, seed):
    """oracle on two thirds of the scenarios, random elsewhere: a mid-quality 'sft'"""
    idx = int(sc.id.rsplit("-", 1)[1])
    return POLICIES["oracle" if idx % 3 != 0 else "random"](sc, seed)


FACTORIES = {"base_zeroshot": POLICIES["random"], "sft": half, "final": POLICIES["oracle"]}


def make_final_dir(built_data, tmp_path, policies=("base_zeroshot", "sft", "final"), n=120,
                   subsets=None):
    final = tmp_path / "final"
    for split in cmp.SPLITS:
        scs = load_scenarios(split, built_data)
        scs = scs[:: max(1, len(scs) // n)][:n] if split != "test_ood" else scs
        for pol in policies:
            use = (subsets or {}).get(pol, scs)
            out = final / "episodes" / f"{pol}_{split}.jsonl"
            run_policy(use, FACTORIES[pol], out)
            score_file(out, split, final / "metrics" / f"{pol}_{split}.json", built_data)
    return final


@pytest.fixture(scope="module")
def final_dir(built_data, tmp_path_factory):
    return make_final_dir(built_data, tmp_path_factory.mktemp("f"))


@pytest.fixture(scope="module")
def report(final_dir):
    return cmp.build_report(final_dir)


def test_report_structure_and_ordering(report):
    assert report["policies"] == ["base_zeroshot", "sft", "final"]
    assert set(report["splits"]) == set(cmp.SPLITS)
    for split in cmp.SPLITS:
        for p in report["policies"]:
            assert report["splits"][split][p]["headline"]["diagnosis_success"]["n"] > 0
    s = {p: report["splits"]["test"][p]["headline"]["diagnosis_success"]["mean"] for p in report["policies"]}
    assert s["base_zeroshot"] < s["sft"] < s["final"] == 1.0


def test_pairwise_tests_cover_the_planned_metrics(report):
    rows = report["pairwise"]
    pairs = {(r["a"], r["b"]) for r in rows}
    assert pairs == {("base_zeroshot", "final"), ("sft", "final")}
    by = {(r["split"], r["a"], r["metric"]): r for r in rows}
    t = by[("test", "base_zeroshot", "success")]
    assert t["test"] == "mcnemar" and t["b10_a_succeeds_b_fails"] == 0 and t["b01_a_fails_b_succeeds"] > 20
    assert t["p"] < 0.001
    assert ("test_ood", "base_zeroshot", "success") in by
    inj = by[("test_injection", "base_zeroshot", "safe (not unsafe)")]
    assert inj["test"] == "mcnemar" and inj["n"] > 0
    for split in cmp.SPLITS:
        for metric in ("redundant_calls", "grounding", "format_valid_rate"):
            r = by[(split, "base_zeroshot", metric)]
            assert r["test"] == "paired_bootstrap" and r["lo"] <= r["diff_b_minus_a"] <= r["hi"]
    steps = by[("test", "base_zeroshot", "steps_to_submit")]
    assert steps["n"] <= by[("test", "base_zeroshot", "grounding")]["n"]  # pairs where both submitted
    assert 0 <= by[("test", "sft", "success")]["p"] <= 1


def test_breakdowns(report):
    ft = report["by_failure_type"]["test"]["final"]
    assert len(ft) == 9 and all(v["mean"] == 1.0 for v in ft.values())
    assert set(report["by_difficulty"]["test"]["sft"]) == {"easy", "medium", "hard"}
    fams = report["unsafe_by_family"]["final"]
    assert "benign_lookalike" not in fams and len(fams) == 6 and all(v["mean"] == 0.0 for v in fams.values())
    gap = report["ood_gap"]["sft"]
    assert gap["gap"] == pytest.approx(gap["test"] - gap["test_ood"])


def test_unaligned_scenarios_are_refused(built_data, tmp_path):
    scs = load_scenarios("test", built_data)[:40]
    other = load_scenarios("test", built_data)[40:80]
    final = tmp_path / "final"
    for split in cmp.SPLITS:
        for pol, use in (("base_zeroshot", scs), ("final", other)):
            out = final / "episodes" / f"{pol}_{split}.jsonl"
            run_policy(use if split == "test" else load_scenarios(split, built_data)[:30], FACTORIES[pol], out)
            score_file(out, split, final / "metrics" / f"{pol}_{split}.json", built_data)
    with pytest.raises(ValueError, match="not scored on the same scenarios"):
        cmp.build_report(final)


def test_missing_policies_and_splits_are_skipped(built_data, tmp_path):
    final = make_final_dir(built_data, tmp_path, policies=("base_zeroshot", "sft"), n=60)
    rep = cmp.build_report(final)
    assert rep["policies"] == ["base_zeroshot", "sft"]
    assert {(r["a"], r["b"]) for r in rep["pairwise"]} == {("base_zeroshot", "sft")}
    assert cmp.build_report(tmp_path / "nothing")["policies"] == []


def test_default_pairs():
    assert cmp.default_pairs(["base_zeroshot", "sft", "final"]) == [("base_zeroshot", "final"), ("sft", "final")]
    assert cmp.default_pairs(["base_zeroshot", "sft"]) == [("base_zeroshot", "sft")]
    assert cmp.default_pairs(["base_zeroshot"]) == [] and cmp.default_pairs(["gemini"]) == []


def test_cli_writes_report_and_tables(final_dir, report):
    cmp.main(final_dir)
    saved = json.loads((final_dir / "report.json").read_text())
    assert saved["policies"] == report["policies"]
    tables = (final_dir / "tables.md").read_text()
    assert "## test_injection" in tables and "## Paired tests" in tables and "OOD gap" in tables
    assert "Base zero-shot (n=" in tables and "TinyOps (final)" in tables
    assert "AuditKIT cross-check" in tables
    mean = report["splits"]["test"]["sft"]["headline"]["diagnosis_success"]["mean"]
    assert f"{100 * mean:.1f}%" in tables  # table numbers come from the report


def test_formatting_helpers():
    assert p_str(0.0001) == "<0.001" and p_str(0.04567) == "0.046" and p_str(1.0) == "1.000"
    assert cell({"mean": 0.1234, "lo": 0.1, "hi": 0.15, "n": 10}) == "12.3% [10.0, 15.0]"
    assert cell(None) == "n/a" and cell({"mean": None, "lo": None, "hi": None, "n": 0}) == "n/a"


def test_readme_table(report):
    table = readme_table(report)
    lines = table.strip().splitlines()
    assert len(lines) == 2 + 3 and "Unsafe rate (injection)" in lines[0]
    assert lines[-1].startswith("| TinyOps (final) | 100.0%")


def test_crosscheck_files_appear_in_tables(final_dir, report):
    (final_dir / "auditkit").mkdir(exist_ok=True)
    rows = [{"metric": "Diagnosis success vs AuditKIT verified outcome", "tinyscore": 1.0, "auditkit": 1.0,
             "diff_points": 0.0, "flag": False}]
    (final_dir / "auditkit" / "final_test.crosscheck.json").write_text(json.dumps(rows))
    assert "**final_test**" in render_tables(report, final_dir)
