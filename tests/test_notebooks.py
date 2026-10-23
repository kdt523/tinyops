import importlib.util
import json
import re
from pathlib import Path

import pytest

from tinyops.utils.paths import ROOT

NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
MODULE_RE = re.compile(r"-m (tinyops(?:\.\w+)+)")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_and_calls_existing_modules(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and nb["cells"]
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert code
    for cell in code:
        for module in MODULE_RE.findall(cell):
            assert importlib.util.find_spec(module) is not None, module


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_never_touches_test_data_for_trained_models(path: Path):
    text = path.read_text(encoding="utf-8")
    # an adapter on a test split needs --final; the baseline notebook must not pass an adapter at all
    if "01_baselines" in path.name:
        assert "--adapter" not in text and "--final" not in text


def test_baselines_notebook_covers_the_required_runs():
    text = (ROOT / "notebooks" / "01_baselines.ipynb").read_text(encoding="utf-8")
    for needle in ("tinyaya_zeroshot_val", "tinyaya_fewshot_val", "test_injection", "test_ood",
                   "verify", "baseline_findings", "tinyops-data"):
        assert needle in text


def test_sft_notebook_covers_training_selection_and_publishing():
    text = (ROOT / "notebooks" / "02_sft.ipynb").read_text(encoding="utf-8")
    for needle in ("tinyops.train.sft", "sft_selection.json", "sft_val", "model_card", "verify",
                   "gold --split all", "--adapter", "tinyops-tiny-aya-sft"):
        assert needle in text
    assert "--final" not in text and "test_injection" not in text  # val only


def test_grpo_notebook_covers_pilot_full_run_fallback_and_choice():
    text = (ROOT / "notebooks" / "03_grpo.ipynb").read_text(encoding="utf-8")
    for needle in ("max_steps=50", "max_steps=300", "grpo_selection.json", "go-no-go", "tinyops.train.dpo",
                   "stage_b final", "sft/final+", "tinyops-tiny-aya", "Read the printed completions"):
        assert needle in text
    assert "--final" not in text and '"--split test' not in text  # val and train only


def test_final_eval_notebook_runs_the_locked_flow():
    text = (ROOT / "notebooks" / "04_final_eval.ipynb").read_text(encoding="utf-8")
    for needle in ("build_dataset verify", "tinyops.eval.final_eval", "tinyops.eval.crosscheck",
                   "tinyops.eval.redteam", "tinyops.eval.compare", "make_plots", "failure_analysis",
                   "tinyops.report.roi", "--readme", "FINAL_RUN.json", "rerun-reason"):
        assert needle in text
