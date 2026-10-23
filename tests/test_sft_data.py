import json
import random

import pytest

from tests.fixtures.fake_tokenizer import FakeTokenizer
from tinyops.policies.parsing import parse_tool_call
from tinyops.policies.prompt import PROMPT_VERSION, build_system_prompt
from tinyops.policies.scripted import OraclePolicy
from tinyops.scenarios.build_dataset import load_scenarios
from tinyops.sim.env import run_episode
from tinyops.train.sft_data import (
    IGNORE,
    MixConfig,
    assistant_spans,
    build_sft_dataset,
    dataset_stats,
    decode_trained,
    episode_row,
    gold_rows,
    mix_rows,
    to_singleturn,
    tokenize_row,
)


@pytest.fixture(scope="module")
def train(built_data):
    scs = load_scenarios("train", built_data)
    return {s.id: s for s in scs}


@pytest.fixture(scope="module")
def gold(train):
    scs = list(train.values())[:120]
    return [run_episode(s.state, OraclePolicy(s.state, random.Random(2)), s.alert_text,
                        scenario_id=s.id) for s in scs]


def test_conversation_matches_inference_and_parses_to_the_oracle_calls(gold, train):
    rows = gold_rows(gold, train)
    assert len(rows) == 120
    for ep, row in zip(gold, rows, strict=True):
        msgs = row["messages"]
        assert msgs[0] == {"role": "system", "content": build_system_prompt()}
        assert msgs[1] == {"role": "user", "content": ep.alert_text}
        assistant = [m for m in msgs if m["role"] == "assistant"]
        assert len(assistant) == len(ep.steps) == len(row["label_turns"])
        for m, step in zip(assistant, ep.steps, strict=True):
            r = parse_tool_call(m["content"])
            assert r.ok and r.format == "fallback_list"
            assert r.call.name == step.tool_name and r.call.arguments == step.arguments
        assert msgs[-1]["role"] == "assistant" and "submit_diagnosis" in msgs[-1]["content"]
        users = [m["content"] for m in msgs[2:] if m["role"] == "user"]
        assert all(u.startswith("Tool result for ") for u in users)


def test_masked_labels_cover_only_assistant_tokens(gold, train):
    tok = FakeTokenizer()
    for ep in gold[:30]:
        row = episode_row(ep, train[ep.scenario_id])
        ex = tokenize_row(tok, row, 100000)
        assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])
        runs = decode_trained(tok, ex)
        expected = [m["content"] + "<|end|>" for m in row["messages"] if m["role"] == "assistant"]
        assert runs == expected  # decode the unmasked tokens: exactly the assistant turns
        text = tok.decode(ex["input_ids"])
        assert text == tok.apply_chat_template(row["messages"])
        assert all(lab == IGNORE for lab, tid in zip(ex["labels"], ex["input_ids"], strict=True)
                   if tok.inv[tid] in ("<|system|>", "<|user|>"))
        assert any(lab != IGNORE for lab in ex["labels"])


def test_system_and_tool_result_tokens_are_never_trained(gold, train):
    tok = FakeTokenizer()
    row = episode_row(gold[0], train[gold[0].scenario_id])
    ex = tokenize_row(tok, row, 100000)
    trained = " ".join(decode_trained(tok, ex))
    assert "on-call SRE" not in trained and "Tool result for" not in trained
    assert row["messages"][1]["content"] not in trained


def test_end_of_turn_is_trained_so_the_model_learns_to_stop(gold, train):
    tok = FakeTokenizer()
    ex = tokenize_row(tok, episode_row(gold[0], train[gold[0].scenario_id]), 100000)
    assert all(run.endswith("<|end|>") for run in decode_trained(tok, ex))


def test_generation_prompt_tokens_are_not_trained(gold, train):
    tok = FakeTokenizer()
    ex = tokenize_row(tok, episode_row(gold[0], train[gold[0].scenario_id]), 100000)
    # "<|assistant|>" (the generation prompt) precedes each trained run and is masked
    for i, (tid, lab) in enumerate(zip(ex["input_ids"], ex["labels"], strict=True)):
        if tok.inv[tid] == "<|assistant|>":
            assert lab == IGNORE and ex["labels"][i + 1] != IGNORE


def test_non_prefix_stable_template_is_rejected(gold, train):
    def bad(messages, add_generation_prompt):
        body = "".join(f"[{m['role']}:{m['content']}]" for m in messages)
        return body + ("\n<bot>" if add_generation_prompt else "")  # gen prompt not in full text

    row = episode_row(gold[0], train[gold[0].scenario_id])
    with pytest.raises(ValueError, match="not prefix-stable"):
        assistant_spans(FakeTokenizer(bad), row["messages"], row["label_turns"])


def test_too_long_examples_are_dropped_and_counted(gold, train):
    tok = FakeTokenizer()
    ds, stats = build_sft_dataset(gold[:20], train, tokenizer=tok, max_seq_len=300)
    assert stats["dropped_too_long"] == 20 and stats["examples"] == 0
    ds, stats = build_sft_dataset(gold[:20], train, tokenizer=tok, max_seq_len=100000)
    assert stats["dropped_too_long"] == 0 and len(ds) == 20 and stats["tokens"] > 0
    assert set(ds.column_names) >= {"input_ids", "attention_mask", "labels", "id"}


def test_chat_dataset_without_tokenizer(gold, train):
    ds, stats = build_sft_dataset(gold[:10], train)
    assert len(ds) == 10 and "messages" in ds.column_names and stats["examples"] == 10
    json.dumps(ds[0]["messages"])


def test_singleturn_mode_has_one_labeled_turn_per_example(gold, train):
    tok = FakeTokenizer()
    ds, _ = build_sft_dataset(gold[:5], train, mode="singleturn", tokenizer=tok, max_seq_len=100000)
    n_calls = sum(len(e.steps) for e in gold[:5])
    assert len(ds) == n_calls
    for ex in ds:
        assert len(decode_trained(tok, ex)) == 1
    rows = to_singleturn(gold_rows(gold[:3], train))
    assert all(r["messages"][-1]["role"] == "assistant" and len(r["label_turns"]) == 1 for r in rows)
    with pytest.raises(ValueError):
        build_sft_dataset(gold[:1], train, mode="weird")


def test_no_val_or_test_scenarios_in_the_training_set(built_data, gold, train):
    val = load_scenarios("val", built_data)
    val_ids = {s.id for s in val}
    rows = gold_rows(gold, train)
    assert not {r["id"] for r in rows} & val_ids
    assert all(r["id"].startswith("train-") and r["split"] == "train" for r in rows)
    test = load_scenarios("test", built_data)[0]
    ep = run_episode(test.state, OraclePolicy(test.state, random.Random(1)), test.alert_text,
                     scenario_id=test.id)
    with pytest.raises(ValueError, match="may not be used for SFT"):
        episode_row(ep, test)


def test_prompt_version_mismatch_is_rejected(gold, train):
    ep = gold[0].model_copy(deep=True)
    ep.meta["prompt_version"] = "v0"
    with pytest.raises(ValueError, match="prompt version"):
        episode_row(ep, train[ep.scenario_id])
    assert gold[0].meta["prompt_version"] == PROMPT_VERSION


def test_injected_share_and_oversampling(train):
    scs = list(train.values())
    eps = [run_episode(s.state, OraclePolicy(s.state, random.Random(2)), s.alert_text,
                       scenario_id=s.id) for s in scs[:400]]
    rows = gold_rows(eps, train)
    base = dataset_stats(rows)
    assert 0.10 <= base["injected_attack_share"] + base["benign_lookalike_share"] <= 0.20
    assert base["benign_lookalike_share"] > 0  # lookalike controls are kept
    doubled = mix_rows(rows, MixConfig(injected_oversample=2.0), seed=1)
    n_attacks = sum(r["injected"] and r["injection_family"] != "benign_lookalike" for r in rows)
    assert len(doubled) == len(rows) + n_attacks
    assert dataset_stats(doubled)["injected_attack_share"] > base["injected_attack_share"]
    assert sum(r["injection_family"] == "benign_lookalike" for r in doubled) == \
        sum(r["injection_family"] == "benign_lookalike" for r in rows)
    assert mix_rows(rows, MixConfig(), 1) != mix_rows(rows, MixConfig(), 2)  # seeded shuffle


def test_teacher_share_is_capped(gold, train):
    rows = gold_rows(gold[:60], train)
    teacher = [{**r, "id": r["id"] + "-t", "source": "teacher"} for r in rows]
    mixed = mix_rows(rows + teacher, MixConfig(max_teacher_share=0.5), seed=3)
    share = dataset_stats(mixed)["teacher_share"]
    assert share <= 0.5 + 1e-9
    few = mix_rows(rows + teacher[:10], MixConfig(max_teacher_share=0.5), seed=3)
    assert dataset_stats(few)["teacher_share"] == pytest.approx(10 / 70)


@pytest.mark.slow
def test_real_hf_tokenizer_offsets_and_template_are_prefix_stable(gold, train):
    """The char-span approach on a real fast tokenizer (SmolLM2 ChatML). Needs network once."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    row = episode_row(gold[0], train[gold[0].scenario_id])
    ex = tokenize_row(tok, row, 100000)
    runs = decode_trained(tok, ex)
    assistant = [m["content"] for m in row["messages"] if m["role"] == "assistant"]
    assert len(runs) == len(assistant)
    for run, content in zip(runs, assistant, strict=True):
        assert content in run and run.rstrip().endswith("<|im_end|>")
