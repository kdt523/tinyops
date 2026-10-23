import csv
import json
import math

import pytest
import yaml

from tinyops.report.roi import (
    break_even_volume,
    compute_roi,
    episodes_per_hour,
    load_roi_config,
    render_markdown,
    token_means,
    training_hours,
)

CFG = {
    "volumes_per_month": [10000, 100000, 1000000],
    "always_on_hours_per_month": 730,
    "closed_model": {"name": "m", "price_per_1m_input_usd": 2.0, "price_per_1m_output_usd": 10.0,
                     "source_url": "https://example.com/p", "date_checked": "2026-10-24"},
    "gpu": {"name": "L4", "price_per_hour_usd": 1.0, "source_url": "https://example.com/g",
            "date_checked": "2026-10-24"},
}
CLOSED = {"prompt_tokens": 10000, "completion_tokens": 1000, "success": 0.80, "unsafe": 0.10}
TINY = {"prompt_tokens": 3000, "completion_tokens": 100, "success": 0.75, "unsafe": 0.02,
        "episodes_per_hour": 100.0}


def test_hand_computed_example():
    roi = compute_roi(CFG, CLOSED, TINY, train_gpu_hours=5.0)
    per_ep = (10000 * 2.0 + 1000 * 10.0) / 1e6
    assert roi["closed_cost_per_episode_usd"] == pytest.approx(per_ep) == pytest.approx(0.03)
    assert roi["one_time_training_cost_usd"] == 5.0
    v10k, v100k, v1m = roi["by_volume"]
    # 10k episodes: 100 GPU-h on demand, but one always-on instance is 730 h -> max() = 730
    assert v10k["gpu_hours_needed"] == 100 and v10k["self_host_on_demand_usd"] == 100.0
    assert v10k["self_host_always_on_usd"] == 730.0 and v10k["self_host_cost_usd"] == 730.0
    assert v10k["closed_cost_usd"] == pytest.approx(300.0) and v10k["months_to_recoup_training"] is None
    # 100k: 1,000 GPU-h -> 2 always-on instances = 1,460 h
    assert v100k["closed_cost_usd"] == pytest.approx(3000.0) and v100k["self_host_cost_usd"] == 1460.0
    assert v100k["monthly_saving_usd"] == pytest.approx(1540.0)
    assert v100k["months_to_recoup_training"] == pytest.approx(5 / 1540)
    # 1M: 10,000 GPU-h -> ceil(10000/730) = 14 instances
    assert v1m["self_host_cost_usd"] == 14 * 730.0
    # cost per successful diagnosis = cost / (volume x success)
    assert v100k["closed_cost_per_success_usd"] == pytest.approx(3000.0 / (100000 * 0.80))
    assert v100k["self_host_cost_per_success_usd"] == pytest.approx(1460.0 / (100000 * 0.75))
    assert roi["quality"]["unsafe"] == {"closed": 0.10, "tinyops": 0.02}


def test_break_even_volume():
    assert break_even_volume(0.03, 1.0, 100.0) == math.ceil(730 / 0.03)  # one instance is enough
    assert break_even_volume(0.009, 1.0, 100.0) is None  # closed is cheaper per episode at any volume
    be = break_even_volume(0.03, 1.0, 100.0)
    closed = be * 0.03
    assert closed >= 730.0 > (be - 1) * 0.03
    roi = compute_roi(CFG, CLOSED, TINY, 0.0)
    assert roi["break_even_monthly_volume"] == be


def test_closed_model_cheaper_case_is_reported_honestly():
    cheap = {**CLOSED, "prompt_tokens": 1000, "completion_tokens": 100}
    roi = compute_roi(CFG, cheap, TINY, 5.0)
    assert roi["break_even_monthly_volume"] is None
    assert all(r["monthly_saving_usd"] < 0 and r["months_to_recoup_training"] is None for r in roi["by_volume"])


def test_markdown_shows_sources_and_quality():
    md = render_markdown(compute_roi(CFG, CLOSED, TINY, 5.0))
    assert "https://example.com/p" in md and "checked 2026-10-24" in md
    assert "| 100,000 |" in md and "diagnosis success 80.0% (closed) vs 75.0% (TinyOps)" in md
    assert "unsafe rate on attacks 10.0% vs 2.0%" in md


def write_cfg(tmp_path, **patch):
    cfg = json.loads(json.dumps(CFG))
    for path, value in patch.items():
        section, key = path.split(".")
        cfg[section][key] = value
    p = tmp_path / "roi.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_missing_price_or_source_raises(tmp_path):
    assert load_roi_config(write_cfg(tmp_path))["gpu"]["price_per_hour_usd"] == 1.0
    for patch in ({"closed_model.price_per_1m_input_usd": None}, {"closed_model.source_url": None},
                  {"closed_model.date_checked": None}, {"gpu.price_per_hour_usd": None},
                  {"gpu.source_url": ""}):
        with pytest.raises(ValueError, match="missing"):
            load_roi_config(write_cfg(tmp_path, **patch))
    with pytest.raises(ValueError, match="must be a URL"):
        load_roi_config(write_cfg(tmp_path, **{"gpu.source_url": "my notes"}))


def test_the_committed_roi_config_refuses_to_run_until_prices_are_cited():
    """configs/roi.yaml leaves the closed-model price empty on purpose (sources disagreed)."""
    with pytest.raises(ValueError, match="closed_model is missing"):
        load_roi_config("configs/roi.yaml")


def test_yaml_dates_become_strings(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("""
volumes_per_month: [10]
always_on_hours_per_month: 730
closed_model: {name: m, price_per_1m_input_usd: 1, price_per_1m_output_usd: 2, source_url: https://a, date_checked: 2026-10-24}
gpu: {name: g, price_per_hour_usd: 1, source_url: https://b, date_checked: 2026-10-24}
""")
    cfg = load_roi_config(p)
    assert cfg["gpu"]["date_checked"] == "2026-10-24"
    json.dumps(cfg)


def test_measured_inputs():
    scores = [{"prompt_tokens": 100, "completion_tokens": 10}, {"prompt_tokens": 300, "completion_tokens": 30},
              {"prompt_tokens": None, "completion_tokens": None}]
    assert token_means(scores) == (200.0, 20.0)
    with pytest.raises(ValueError, match="no token usage"):
        token_means([{"prompt_tokens": None, "completion_tokens": None}])
    man = [{"new_episodes_this_invocation": 270, "elapsed_s_this_invocation": 1800},
           {"new_episodes_this_invocation": 162, "elapsed_s_this_invocation": 1800}]
    assert episodes_per_hour(man) == pytest.approx(432 / 1.0)
    with pytest.raises(ValueError):
        episodes_per_hour([{"new_episodes_this_invocation": 0, "elapsed_s_this_invocation": 10}])


def test_training_hours_sums_only_training_runs(tmp_path):
    log = tmp_path / "compute_log.csv"
    with log.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "run", "gpu", "hours"])
        w.writerows([["d", "sft_train", "T4", "1.5"], ["d", "grpo_train", "T4", "2.0"],
                     ["d", "tinyaya_zeroshot_val", "T4", "0.4"], ["d", "dpo_train", "T4", "0.5"]])
    assert training_hours(log, ["sft_train", "grpo_train", "dpo_train"]) == 4.0
    assert training_hours(tmp_path / "missing.csv", ["sft_train"]) == 0.0
