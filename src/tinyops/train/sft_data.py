"""SFT data: gold episodes -> chat conversations identical to inference -> masked token ids.

Loss is computed on assistant turns only. Masks are built from the chat template itself: the
characters an assistant turn adds to the rendered conversation are exactly the span to train on
(including the end-of-turn token, so the model learns to stop).
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from tinyops.policies.prompt import PROMPT_VERSION
from tinyops.scenarios.base import Scenario
from tinyops.sim.env import Episode
from tinyops.sim.export import conversation_from_episode
from tinyops.utils.log import get_logger

log = get_logger(__name__)

IGNORE = -100
TRAIN_SPLITS = ("train", "train_extra")


@dataclass
class MixConfig:
    injected_oversample: float = 1.0  # extra copies of attack-injected episodes (2.0 = double)
    max_teacher_share: float = 0.5


def episode_row(ep: Episode, sc: Scenario, source: str = "gold") -> dict:
    """One training conversation. Refuses anything that is not train data or not prompt v1."""
    if sc.split not in TRAIN_SPLITS and sc.split != "val":
        raise ValueError(f"{sc.id}: {sc.split} data may not be used for SFT")
    if ep.meta.get("prompt_version") != PROMPT_VERSION:
        raise ValueError(f"{ep.scenario_id}: prompt version {ep.meta.get('prompt_version')!r} "
                         f"!= {PROMPT_VERSION!r}")
    messages = conversation_from_episode(ep)
    return {
        "id": ep.scenario_id, "messages": messages,
        "label_turns": [i for i, m in enumerate(messages) if m["role"] == "assistant"],
        "failure_type": sc.failure_type, "difficulty": sc.difficulty, "split": sc.split,
        "injected": sc.injected, "injection_family": sc.injection_family, "source": source,
    }


def gold_rows(episodes: list[Episode], scenarios: dict[str, Scenario], source: str = "gold") -> list[dict]:
    return [episode_row(e, scenarios[e.scenario_id], source) for e in episodes]


def to_singleturn(rows: list[dict]) -> list[dict]:
    """Each assistant turn becomes its own example (prompt = everything before it)."""
    out = []
    for r in rows:
        for i in r["label_turns"]:
            out.append({**r, "id": f"{r['id']}#{i}", "messages": r["messages"][: i + 1],
                        "label_turns": [i]})
    return out


def mix_rows(rows: list[dict], mix: MixConfig, seed: int = 1234) -> list[dict]:
    """Oversample attack-injected episodes if asked; benign lookalikes are always kept."""
    rng = random.Random(seed)
    out = list(rows)
    extra = mix.injected_oversample - 1.0
    if extra > 0:
        attacks = [r for r in rows if r["injected"] and r["injection_family"] != "benign_lookalike"]
        for _ in range(int(extra)):
            out += attacks
        out += [r for r in attacks if rng.random() < extra - int(extra)]
    teacher = [r for r in out if r["source"] == "teacher"]
    gold = [r for r in out if r["source"] != "teacher"]
    cap = int(mix.max_teacher_share * len(out))
    if len(teacher) > cap:
        # keep share <= max: teacher <= share/(1-share) * gold
        keep = int(mix.max_teacher_share / (1 - mix.max_teacher_share) * len(gold))
        teacher = rng.sample(teacher, min(keep, len(teacher)))
    out = gold + teacher
    rng.shuffle(out)
    return out


def assistant_spans(tokenizer: Any, messages: list[dict], label_turns: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """(rendered text, character spans of the assistant turns to train on)."""
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    spans = []
    for i in label_turns:
        prefix = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True)
        upto = tokenizer.apply_chat_template(messages[: i + 1], tokenize=False)
        if not (upto.startswith(prefix) and text.startswith(upto)):
            raise ValueError(
                "chat template is not prefix-stable: cannot derive the assistant span of turn "
                f"{i}. Decode an example and compare with inference.\n--- prefix ---\n"
                f"{prefix[-200:]}\n--- turn ---\n{upto[len(prefix) - 50:][:300]}")
        spans.append((len(prefix), len(upto)))
    return text, spans


def tokenize_row(tokenizer: Any, row: dict, max_seq_len: int) -> dict | None:
    """input_ids / attention_mask / labels (-100 outside assistant turns); None if too long."""
    text, spans = assistant_spans(tokenizer, row["messages"], row["label_turns"])
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_seq_len:
        return None
    labels = []
    for tid, (s, e) in zip(ids, offsets, strict=True):
        inside = e > s and any(s >= a and e <= b for a, b in spans)
        labels.append(tid if inside else IGNORE)
    return {"input_ids": list(ids), "attention_mask": [1] * len(ids), "labels": labels}


def decode_trained(tokenizer: Any, example: dict) -> list[str]:
    """The text of each contiguous run of trained tokens (for eyeballing and tests)."""
    runs, cur = [], []
    for tid, lab in zip(example["input_ids"], example["labels"], strict=True):
        if lab != IGNORE:
            cur.append(tid)
        elif cur:
            runs.append(tokenizer.decode(cur))
            cur = []
    if cur:
        runs.append(tokenizer.decode(cur))
    return runs


def dataset_stats(rows: list[dict], n_tokens: int | None = None, dropped: int = 0) -> dict:
    n = len(rows)
    attacks = [r for r in rows if r["injected"] and r["injection_family"] != "benign_lookalike"]
    benign = [r for r in rows if r["injection_family"] == "benign_lookalike"]
    teacher = [r for r in rows if r["source"] == "teacher"]
    return {
        "examples": n, "tokens": n_tokens, "dropped_too_long": dropped,
        "dropped_share": dropped / (n + dropped) if n + dropped else 0.0,
        "injected_attack_share": len(attacks) / n if n else 0.0,
        "benign_lookalike_share": len(benign) / n if n else 0.0,
        "teacher_share": len(teacher) / n if n else 0.0,
        "by_failure_type": dict(sorted(Counter(r["failure_type"] for r in rows).items())),
        "by_source": dict(sorted(Counter(r["source"] for r in rows).items())),
    }


def build_sft_dataset(
    gold_episodes: list[Episode],
    scenarios: dict[str, Scenario],
    mode: str = "multiturn",
    *,
    tokenizer: Any = None,
    max_seq_len: int = 4096,
    mix: MixConfig | None = None,
    extra_rows: list[dict] | None = None,
    seed: int = 1234,
):
    """HF `datasets.Dataset`. Without a tokenizer: chat rows (`messages`, `label_turns`, ...).
    With one: tokenized rows (`input_ids`, `attention_mask`, `labels`) ready for SFTTrainer.
    Returns (dataset, stats)."""
    from datasets import Dataset

    if mode not in ("multiturn", "singleturn"):
        raise ValueError(f"mode must be multiturn or singleturn, got {mode!r}")
    rows = gold_rows(gold_episodes, scenarios) + (extra_rows or [])
    rows = mix_rows(rows, mix or MixConfig(), seed)
    if mode == "singleturn":
        rows = to_singleturn(rows)
    if tokenizer is None:
        return Dataset.from_list(rows), dataset_stats(rows)
    kept, tokenized, dropped = [], [], 0
    for r in rows:
        t = tokenize_row(tokenizer, r, max_seq_len)
        if t is None:
            dropped += 1
            continue
        kept.append(r)
        tokenized.append({**t, "id": r["id"]})
    if dropped:
        log.warning("dropped %d/%d examples longer than %d tokens", dropped, len(rows), max_seq_len)
    n_tokens = sum(len(t["input_ids"]) for t in tokenized)
    return Dataset.from_list(tokenized), dataset_stats(kept, n_tokens, dropped)
