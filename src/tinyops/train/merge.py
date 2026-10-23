"""Merge a LoRA adapter into its base model (Stage B trains on top of the merged SFT model)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tinyops.utils.log import get_logger

log = get_logger(__name__)


def merge_adapter(base: str, adapter: str, out_dir: Path | str, dtype: str = "auto") -> Path:
    """Write base+adapter as a standalone model (weights + tokenizer) to `out_dir`."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(out_dir)
    if (out / "config.json").exists():
        log.info("merged model already at %s", out)
        return out
    if dtype == "auto":
        cuda = torch.cuda.is_available()
        dt = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
    else:
        dt = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dt)
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    log.info("merged %s + %s -> %s", base, adapter, out)
    return out


def main(
    base: Annotated[str, typer.Option()],
    adapter: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
    dtype: Annotated[str, typer.Option()] = "auto",
) -> None:
    merge_adapter(base, adapter, out, dtype)


if __name__ == "__main__":
    typer.run(main)
