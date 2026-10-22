---
name: sft-training
description: Skill 6 of 9 for TinyOps. Use to turn gold trajectories into training-identical multi-turn chat data and train a LoRA SFT adapter on Tiny Aya with AgentTune's create_distill_trainer (or TRL SFTTrainer fallback), with val-based checkpoint selection under a safety constraint.
---

# Skill 6 — Stage A: SFT on gold trajectories

## Goal
A LoRA adapter that makes Tiny Aya reliably emit valid tool calls, investigate sensibly, ignore injected instructions, and submit grounded diagnoses — measured on val.

## Inputs
- `data/train|val/gold_episodes.jsonl` (skill 3), parser + eval runner (skill 5), `docs/BASELINE_FINDINGS.md`.

## Files
```
src/tinyops/train/sft_data.py
src/tinyops/train/sft.py
src/tinyops/train/teacher_data.py      # optional
configs/sft.yaml
notebooks/02_sft.ipynb
tests/test_sft_data.py
```

## 6.1 Data — `train/sft_data.py`
`build_sft_dataset(gold_episodes, scenarios, mode="multiturn") -> datasets.Dataset`
- Each example = the **full conversation** exactly as inference sees it: system prompt (`build_system_prompt`, same `PROMPT_VERSION`) → user alert → for each step: assistant turn = tool call in the fallback format (exact string the oracle emitted) → user turn = tool result in `tool_result_format` → … → final assistant turn = the `submit_diagnosis` call.
- **Loss masking:** compute loss only on assistant turns. If using TRL `SFTTrainer`, use `assistant_only_loss=True` if the installed version supports it for this template; otherwise build `labels` manually (tokenize turn by turn, `-100` on non-assistant tokens). Verify by decoding the unmasked tokens of one example in a test.
- Use the tokenizer's chat template (`apply_chat_template(..., tokenize=False)`) so special tokens match inference.
- Length: drop/truncate examples whose tokenized length > `max_seq_len` (default 3072); log how many (should be < 2%). Hard scenarios with long logs are the usual culprits — the oracle already uses a moderate `tail`.
- Mix (from `configs/sft.yaml`): all train gold episodes; injected share ≈ 15–20% (oversample injected ones if the baseline showed injection obedience); include the `benign_lookalike` controls so the model doesn't learn to refuse everything with "delete" in it.
- Optional teacher data (`train/teacher_data.py`): run Gemini (`APIPolicy`) on extra *train-split* scenarios generated with a new seed (`--seed 5678`, written to `data/train_extra/`), keep only episodes with `success=True`, `unsafe=False`, `redundant_calls=0`, `grounding=1.0`. Cap teacher share at 50%. Record counts.
- Also produce `mode="singleturn"` (each assistant turn as a separate prompt→completion pair) in case `create_distill_trainer` only supports pairs.
- Save to `data/sft/{train,val}` (HF `datasets` format) + a stats JSON (examples, tokens, mix shares).

## 6.2 Trainer — `train/sft.py`
CLI: `python -m tinyops.train.sft --config configs/sft.yaml`
1. **Primary path:** AgentTune `create_distill_trainer(student="CohereLabs/tiny-aya-global", trajectories=… | demonstrations=…, output_dir=…, …)`. Before using it, re-read its source: does it accept multi-turn tool conversations with assistant-only loss? Does it take a `peft_config` and TRL args?
2. **Fallback path** (if 1 can't do multi-turn assistant-only loss cleanly): TRL `SFTTrainer` + `peft.LoraConfig` directly on the dataset from 6.1. Log the reason in DECISIONS.md. Either way, write `lexsi_provenance.json` the same way AgentTune does (library, version, base model, method, dataset) — copy AgentTune's helper if public.
3. Evaluation callback: at the end of each epoch (and every `eval_every_steps`), run the eval runner on a fixed **val subset of 90 scenarios** (10/type, includes injected) with greedy decoding, compute TinyScore, log `val/diagnosis_success`, `val/format_valid_rate`, `val/unsafe_rate`, `val/redundant_calls_per_ep`. Save adapter checkpoints.
4. **Checkpoint selection:** highest `val/diagnosis_success` among checkpoints whose `val/unsafe_rate` ≤ base model's val unsafe rate (and ≤ 5% absolute). Write the choice to `results/runs/sft_selection.json`.
5. After selection, run the full val (180) once with the selected adapter → `results/metrics/sft_val.json`.

`configs/sft.yaml` (starting point; tune only on val):
```yaml
base_model: CohereLabs/tiny-aya-global
lora: {r: 16, alpha: 32, dropout: 0.05, target_modules: all-linear}
lr: 1.0e-4
scheduler: cosine
warmup_ratio: 0.03
epochs: 2
max_seq_len: 3072
per_device_batch: 1          # T4; 2 on L4/A10G
grad_accum: 16
gradient_checkpointing: true
dtype: auto                  # fp16 on T4, bf16 on L4+
eval_every_steps: 100
val_subset: 90
seed: 1234
push_to_hub: <user>/tinyops-tiny-aya-sft
```
Memory notes: 3.35B fp16 ≈ 6.7 GB weights + LoRA + activations fits a 16 GB T4 with batch 1, checkpointing on, seq 3072. If OOM: seq 2048, or QLoRA (4-bit base) — record which.

## 6.3 Notebook — `notebooks/02_sft.ipynb`
GPU check → install → envinfo → data fetch + hash check → build SFT dataset → train (resumable from last checkpoint; push checkpoints to the Hub every save) → selection → full val eval → push adapter + model card draft + `lexsi_provenance.json` → append GPU hours to `compute_log.csv`.

## 6.4 Model card (draft, finalized in skill 9)
Base model + licence (CC-BY-NC), data (generated KubeSim scenarios, counts, teacher share), method + hyperparameters, val metrics (from files), intended use (research; simulated environment), limitations, how to load with PEFT.

## 6.5 Decision rules
- If SFT val `unsafe_rate` > base: add more injected-and-ignored examples (oversample ×2), retrain once.
- If SFT val success < few-shot baseline: check loss masking and template mismatch first (decode an example), then lr (try 5e-5).
- If format validity < 95%: inspect parse errors; usually a template/format string mismatch between training data and inference.
- Time box: if SFT isn't working by **Nov 7 EOD**, ship the best checkpoint and continue to skill 7's DPO path.

## Tests (CPU)
- `test_sft_data.py`: a gold episode renders to a conversation whose assistant turns parse with `parse_tool_call` and match the oracle calls; masked labels cover only assistant tokens (decode and compare); system prompt version matches inference; no val/test scenario ids in the train set; injected share within configured bounds.

## Exit gate
- [ ] SFT adapter pushed to `<user>/tinyops-tiny-aya-sft` with provenance file.
- [ ] `results/metrics/sft_val.json` shows higher diagnosis success and format validity than zero-shot **and** few-shot on val, with unsafe rate not higher than base.
- [ ] Selection rationale saved; DECISIONS.md updated.
- [ ] Commit `skill-6: SFT data pipeline and adapter`.
