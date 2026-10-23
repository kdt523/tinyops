import json
import re
import shutil

import pytest

from tests.test_compare import make_final_dir
from tinyops.eval import compare as cmp
from tinyops.report import article, make_replays
from tinyops.report import check_submission as cs
from tinyops.report.make_tables import readme_table, update_readme
from tinyops.utils.io import sha256_file, write_json
from tinyops.utils.paths import ROOT


@pytest.fixture(scope="module")
def final_dir(built_data, tmp_path_factory):
    d = make_final_dir(built_data, tmp_path_factory.mktemp("sub"))
    cmp.main(d)
    return d


# -- article ---------------------------------------------------------------------------------------------
def test_article_without_results_has_markers_and_no_percentages(tmp_path, built_data):
    text = article.render_article(ROOT, tmp_path, built_data)
    assert len(re.findall(r"^## \d+\. ", text, re.MULTILINE)) == 12
    assert text.count("⟨pending") >= 5
    assert not re.search(r"\d+(\.\d+)?\s?%", text), re.findall(r".{20}\d+(?:\.\d+)?\s?%.{10}", text)
    assert "train: 1215" in text and "test_injection: 162" in text  # data facts, read from stats.json
    for f in ("crashloop_missing_env", "benign_lookalike"):
        assert f in text


def test_article_with_results_quotes_the_report(final_dir, built_data, tmp_path):
    res = tmp_path / "res"
    shutil.copytree(final_dir, res / "final")
    (res / "final" / "roi.md").write_text("# Cost comparison (monthly)\n\n| Episodes / month | x |\n|---|---|\n| 10,000 | $1 |\n")
    report = json.loads((final_dir / "report.json").read_text())
    text = article.render_article(ROOT, res, built_data)
    a = report["splits"]["test"]["base_zeroshot"]["headline"]["diagnosis_success"]["mean"]
    b = report["splits"]["test"]["final"]["headline"]["diagnosis_success"]["mean"]
    tl = text.split("## 2.")[0]
    assert f"{100 * a:.1f}%" in tl and f"{100 * b:.1f}%" in tl and "(base Tiny Aya)" in tl
    assert "Unsafe action rate" in tl and "Redundant tool calls" in tl
    assert "| Episodes / month |" in text.split("## 9. Cost")[1].split("## 10.")[0]
    assert "Paired tests" not in text.split("## 6. Results")[0]
    assert text.split("## 6. Results")[1].count("McNemar") >= 1 or "mcnemar" in text.lower()


def test_article_failure_section_comes_from_the_confirmed_doc(built_data, tmp_path):
    root = tmp_path / "root"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "FAILURE_ANALYSIS.md").write_text(
        "# Failure analysis (final model)\n\n> **STATUS: CONFIRMED**\n\n| Category | Count |\n|---|---|\n| wrong tool path | 3 |\n\n## Illustrative examples\n")
    text = article.render_article(root, tmp_path / "none", built_data)
    sec = text.split("## 8. What still breaks")[1].split("## 9.")[0]
    assert "STATUS: CONFIRMED" in sec and "| wrong tool path | 3 |" in sec


# -- README -----------------------------------------------------------------------------------------------
def test_update_readme_replaces_only_the_block(final_dir):
    report = json.loads((final_dir / "report.json").read_text())
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    new = update_readme(readme, report)
    block = new.split("<!-- RESULTS:START -->\n")[1].split("\n<!-- RESULTS:END -->")[0]
    assert block.startswith(readme_table(report).strip()) and "Pending" not in block
    assert new.split("<!-- RESULTS:START -->")[0] == readme.split("<!-- RESULTS:START -->")[0]
    assert new.split("<!-- RESULTS:END -->")[1] == readme.split("<!-- RESULTS:END -->")[1]
    assert update_readme(new, report) == new  # idempotent
    with pytest.raises(ValueError, match="no <!-- RESULTS"):
        update_readme("# nothing", report)


def test_committed_readme_quickstart_commands_exist():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    import importlib.util

    for module in set(re.findall(r"python -m (tinyops[\w.]+)", text)):
        assert importlib.util.find_spec(module), module


# -- submission checker --------------------------------------------------------------------------------------
@pytest.fixture
def repo(final_dir, built_data, tmp_path):
    root = tmp_path / "repo"
    res = root / "results"
    shutil.copytree(final_dir, res / "final", ignore=shutil.ignore_patterns("plots", "failures", "report.json"))
    shutil.copy(final_dir / "report.json", res / "final" / "report.json")
    for rel in ("space/app.py", "space/replay_view.py", "space/requirements.txt", "space/README.md"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, root / rel)
    make_replays.build_replays(final_dir, root / "space" / "replays", n=12, seed=1, data=built_data)
    shutil.copy(final_dir / "report.json", root / "space" / "report.json")
    report = json.loads((final_dir / "report.json").read_text())
    docs = {
        "README.md": "# T\n<!-- RESULTS:START -->\n" + readme_table(report) + "\n<!-- RESULTS:END -->\n",
        "LICENSE": "MIT License\n", "docs/ARTICLE.md": "# A\n| Episodes / month | x |\n",
        "docs/DEMO_SCRIPT.md": "d", "docs/LEXSI_FEEDBACK.md": "d", "docs/METRICS.md": "d", "docs/DECISIONS.md": "d",
        "docs/BASELINE_FINDINGS.md": "d", "docs/FAILURE_ANALYSIS.md": "> **STATUS: CONFIRMED** — 3 of 3",
        "hub/model_card_final.md": "card <user>", "hub/dataset_card.md": "card <user>", "data/TEST_HASHES.txt": "x",
        ".gitignore": ".env\n", "results/final/roi.json": "{}",
    }
    for rel, text in docs.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    write_json(res / "final" / "FINAL_RUN.json", {
        "missing": [], "episodes_sha256": {str(p.relative_to(res / "final")): sha256_file(p)
                                           for p in sorted((res / "final" / "episodes").glob("*.jsonl"))}})
    return root, res, built_data


def status(checks):
    return {c.name: c.status for c in checks}


def test_a_complete_repo_passes_every_automatic_check(repo):
    root, res, data = repo
    checks = cs.run_checks(root, res, data)
    auto = [c for c in checks if c.status != cs.MANUAL]
    assert all(c.status == cs.PASS for c in auto), [(c.name, c.status, c.detail) for c in auto if c.status != cs.PASS]
    assert sum(c.status == cs.MANUAL for c in checks) == len(cs.MANUAL_ITEMS) >= 8
    assert "[ ] you" in cs.render(checks)


@pytest.mark.parametrize(("breakage", "check"), [
    ("placeholder", "no leftover placeholders / pending markers"),
    ("pending", "no leftover placeholders / pending markers"),
    ("secret", "no secrets in the repository"),
    ("draft", "failure analysis confirmed by a human"),
    ("tamper", "final run is locked (FINAL_RUN.json)"),
    ("readme", "README results block"),
    ("file", "required files exist"),
    ("replays", "Space replay mode has data"),
    ("hashes", "locked test splits match TEST_HASHES.txt"),
])
def test_each_breakage_is_caught(repo, tmp_path, breakage, check):
    root, res, data = repo
    if breakage == "placeholder":
        (root / "README.md").write_text((root / "README.md").read_text() + "\nhttps://github.com/<github-user>/tinyops\n")
    elif breakage == "pending":
        (root / "docs" / "ARTICLE.md").write_text("x")
        (root / "README.md").write_text((root / "README.md").read_text() + "\n⟨pending: x⟩\n")
    elif breakage == "secret":
        (root / "docs" / "DECISIONS.md").write_text("key hf_" + "b" * 30)
    elif breakage == "draft":
        (root / "docs" / "FAILURE_ANALYSIS.md").write_text("> **STATUS: DRAFT**")
    elif breakage == "tamper":
        with open(res / "final" / "episodes" / "final_test.jsonl", "a") as f:
            f.write("\n")
    elif breakage == "readme":
        (root / "README.md").write_text("<!-- RESULTS:START -->\n_Pending: nothing_\n<!-- RESULTS:END -->")
    elif breakage == "file":
        (root / "docs" / "DEMO_SCRIPT.md").unlink()
    elif breakage == "replays":
        for p in (root / "space" / "replays").glob("test-*.json"):
            p.unlink()
    elif breakage == "hashes":
        shutil.copytree(data, tmp_path / "d2")
        with open(tmp_path / "d2" / "test_ood" / "scenarios.jsonl", "a") as f:
            f.write(" ")
        data = tmp_path / "d2"
    got = status(cs.run_checks(root, res, data))
    assert got[check] == cs.FAIL, got


def test_user_placeholder_is_allowed_in_cards_but_not_in_the_readme(repo):
    root, res, data = repo
    assert status(cs.run_checks(root, res, data))["no leftover placeholders / pending markers"] == cs.PASS
    (root / "README.md").write_text((root / "README.md").read_text() + "\nmodel <user>/x\n")
    assert status(cs.run_checks(root, res, data))["no leftover placeholders / pending markers"] == cs.FAIL


def test_roi_is_optional_so_it_only_warns(repo):
    root, res, data = repo
    (root / "docs" / "ARTICLE.md").write_text("# A")
    assert status(cs.run_checks(root, res, data))["ROI section in the article (optional, preferred)"] == cs.WARN


def test_before_after_needs_cis_for_two_metrics(repo):
    root, res, data = repo
    rep = json.loads((res / "final" / "report.json").read_text())
    rep["splits"]["test_injection"]["final"]["headline"]["unsafe_rate"]["lo"] = None
    rep["splits"]["test"]["final"]["headline"]["format_valid_rate"]["lo"] = None
    (res / "final" / "report.json").write_text(json.dumps(rep))
    assert status(cs.run_checks(root, res, data))[">=2 before/after metrics with CIs"] == cs.FAIL


def test_the_real_repo_currently_fails_on_missing_results_not_on_code(tmp_path):
    """Today the project has no final results: the checker must say so (and not crash)."""
    checks = cs.run_checks(ROOT, tmp_path / "no_results", ROOT / "data")
    got = status(checks)
    assert got["required files exist"] == cs.FAIL and got["final run is locked (FINAL_RUN.json)"] == cs.FAIL
    assert got["locked test splits match TEST_HASHES.txt"] == cs.PASS
    assert got["no secrets in the repository"] == cs.PASS
