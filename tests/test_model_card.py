from tinyops.train.model_card import PENDING, render_sft_card
from tinyops.train.sft import load_config


def metrics(v):
    stat = {"mean": v, "lo": v - 0.05, "hi": v + 0.05, "n": 180}
    return {"headline": {k: stat for k in ("diagnosis_success", "format_valid_rate",
                                           "schema_valid_rate", "unsafe_rate",
                                           "redundant_calls_per_ep")}}


def test_card_shows_numbers_only_from_the_given_files():
    cfg = load_config("configs/sft.yaml")
    stats = {"examples": 1215, "tokens": 3228584, "injected_attack_share": 0.12,
             "benign_lookalike_share": 0.03, "teacher_share": 0.0}
    sel = {"chosen": {"step": 100, "diagnosis_success": 0.61}, "constraint_met": True}
    card = render_sft_card(cfg, stats, sel, metrics(0.7), metrics(0.2), None)
    assert "70.0% [65.0, 75.0]" in card and "20.0% [15.0, 25.0]" in card
    assert card.count(PENDING) == 5  # the few-shot column has no file yet: stated, not invented
    assert "step 100" in card and "1215 multi-turn gold" in card and "CC-BY-NC" in card
    assert "LSAL v1.1" in card and "base_model: CohereLabs/tiny-aya-global" in card


def test_card_with_no_results_claims_nothing():
    card = render_sft_card(load_config("configs/sft.yaml"), None, None, None, None, None)
    assert PENDING in card and "%" not in card.split("## Evaluation")[1].split("Metrics are")[0].replace("|---", "")
