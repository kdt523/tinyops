"""A tiny offline tokenizer with a prefix-stable chat template (for mask/label tests)."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"<\|\w+\|>|\w+|\s+|[^\w\s]")


class FakeTokenizer:
    def __init__(self, template=None):
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}
        self.template = template or self._chatml

    @staticmethod
    def _chatml(messages, add_generation_prompt=False):
        text = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        return text + ("<|assistant|>" if add_generation_prompt else "")

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        return self.template(messages, add_generation_prompt)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offs = [], []
        for m in _TOKEN.finditer(text):
            tok = m.group(0)
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab) + 1
                self.inv[self.vocab[tok]] = tok
            ids.append(self.vocab[tok])
            offs.append((m.start(), m.end()))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out

    def decode(self, ids):
        return "".join(self.inv[i] for i in ids)
