"""Hugging Face transformers policy (Tiny Aya and others), greedy, batched, optional LoRA adapter."""

from __future__ import annotations

import time
from typing import Any

from tinyops.policies.base import PolicyOutput
from tinyops.policies.parsing import has_complete_call_json, parse_tool_call
from tinyops.sim.env import Episode
from tinyops.sim.export import conversation_from_episode
from tinyops.utils.log import get_logger

log = get_logger(__name__)


def fewshot_messages(episodes: list[Episode]) -> list[dict]:
    """Prior conversation turns (alert, calls, results ...) of each example, without the system."""
    out: list[dict] = []
    for ep in episodes:
        out += conversation_from_episode(ep)[1:]
    return out


def split_adapters(spec: str) -> list[str]:
    """`a+b` -> ["a", "b"] (a path or hub id never contains '+')."""
    parts = [p.strip() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"empty adapter spec {spec!r}")
    return parts


def splice_fewshot(messages: list[dict], shots: list[dict]) -> list[dict]:
    """[system] + few-shot turns + the real conversation."""
    if not shots:
        return messages
    return messages[:1] + shots + messages[1:]


class HFPolicy:
    def __init__(
        self,
        model_id: str,
        adapter: str | None = None,
        dtype: str = "auto",
        load_in_4bit: bool = False,
        max_new_tokens: int = 256,
        fewshot: list[Episode] | None = None,
        name: str | None = None,
    ) -> None:
        self.model_id, self.adapter, self.dtype = model_id, adapter, dtype
        self.load_in_4bit, self.max_new_tokens = load_in_4bit, max_new_tokens
        self.fewshot = fewshot or []
        self._shots = fewshot_messages(self.fewshot)
        base = model_id.rsplit("/", 1)[-1]
        self.name = name or "-".join(
            [base] + ([split_adapters(adapter)[-1].rstrip("/").rsplit("/", 1)[-1]] if adapter else [])
            + ([f"fewshot{len(self.fewshot)}"] if self.fewshot else ["zeroshot"])
        )
        self._model: Any = None
        self._tok: Any = None
        self._resolved_dtype: str | None = None

    @classmethod
    def from_loaded(cls, model: Any, tokenizer: Any, name: str, max_new_tokens: int = 256) -> HFPolicy:
        """Wrap an already loaded (possibly mid-training) model; nothing is downloaded."""
        pol = cls(getattr(model, "name_or_path", None) or "loaded-model", name=name,
                  max_new_tokens=max_new_tokens)
        pol._model, pol._tok = model, tokenizer
        pol._resolved_dtype = str(getattr(model, "dtype", "unknown")).replace("torch.", "")
        return pol

    # -- loading --------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cuda = torch.cuda.is_available()
        if self.dtype == "auto":
            if cuda:
                dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dt = torch.float32
        else:
            dt = getattr(torch, self.dtype)
        self._resolved_dtype = str(dt).replace("torch.", "")
        tok = AutoTokenizer.from_pretrained(self.model_id)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kwargs: dict[str, Any] = {}
        kwargs["dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"] = dt
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt)
        if cuda:
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if self.adapter:
            from peft import PeftModel

            # "sft_adapter+grpo_adapter": earlier adapters are merged in order (Stage B was trained
            # on the merged SFT model); the last stays an unmerged adapter
            *merge, last = split_adapters(self.adapter)
            for a in merge:
                model = PeftModel.from_pretrained(model, a).merge_and_unload()
            model = PeftModel.from_pretrained(model, last)
        model.eval()
        self._model, self._tok = model, tok
        log.info("loaded %s (adapter=%s, dtype=%s, 4bit=%s)", self.model_id, self.adapter,
                 self._resolved_dtype, self.load_in_4bit)

    def config(self) -> dict:
        return {
            "policy": "hf", "name": self.name, "model_id": self.model_id, "adapter": self.adapter,
            "dtype": self._resolved_dtype or self.dtype, "load_in_4bit": self.load_in_4bit,
            "max_new_tokens": self.max_new_tokens, "fewshot": len(self.fewshot),
            "decoding": "greedy", "stop": "complete JSON call or EOS",
        }

    # -- generation -----------------------------------------------------
    def _stopper(self, prompt_len: int):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        tok = self._tok

        class JsonComplete(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                done = [
                    has_complete_call_json(tok.decode(row[prompt_len:], skip_special_tokens=True))
                    for row in input_ids
                ]
                return torch.tensor(done, dtype=torch.bool, device=input_ids.device)

        return StoppingCriteriaList([JsonComplete()])

    def act_batch(self, batch: list[list[dict]]) -> list[PolicyOutput]:
        """Greedy generation; on CUDA OOM the batch is halved and retried (never fails episodes)."""
        import torch

        try:
            return self._generate(batch)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            log.warning("CUDA OOM at batch size %d; retrying as two halves", len(batch))
            torch.cuda.empty_cache()
            mid = len(batch) // 2
            return self.act_batch(batch[:mid]) + self.act_batch(batch[mid:])

    def _generate(self, batch: list[list[dict]]) -> list[PolicyOutput]:
        import torch

        self._load()
        tok, model = self._tok, self._model
        texts = [
            tok.apply_chat_template(splice_fewshot(m, self._shots), add_generation_prompt=True,
                                    tokenize=False)
            for m in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False, temperature=None,
                top_p=None, top_k=None, pad_token_id=tok.pad_token_id,
                stopping_criteria=self._stopper(prompt_len),
            )
        elapsed = time.perf_counter() - t0
        results = []
        for i in range(len(batch)):
            gen = out[i, prompt_len:]
            text = tok.decode(gen, skip_special_tokens=True)
            parsed = parse_tool_call(text)
            results.append(PolicyOutput(
                text=text, tool_call=parsed.call,
                usage={
                    "prompt_tokens": int(enc["attention_mask"][i].sum()),
                    "completion_tokens": int((gen != tok.pad_token_id).sum()),
                    "latency_s": elapsed / len(batch), "batch_latency_s": elapsed,
                    "batch_size": len(batch),
                }))
        return results

    def act(self, messages: list[dict]) -> PolicyOutput:
        return self.act_batch([messages])[0]
