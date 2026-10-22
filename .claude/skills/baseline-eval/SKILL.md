---
name: baseline-eval
description: Skill 5 of 9 for TinyOps. Use to implement model policies (Tiny Aya via transformers, closed/teacher model via LiteLLM), the robust tool-call parser, the batched resumable eval runner, and the Colab/Kaggle notebook that produces honest "before" numbers and the top failure modes.
---

# Skill 5 — Policies, parser, eval runner & baselines

## Goal
Honest baseline ("before") numbers for Tiny Aya on val and the test splits, a closed-model reference, and a written list of the base model's top failure modes that drives reward design.

## Inputs
- Skills 2–4. A GPU session (Colab T4 / Kaggle / HF credits). HF token with Tiny Aya access. Gemini API key.

## Files
```
src/tinyops/policies/parsing.py
src/tinyops/policies/hf_policy.py
src/tinyops/policies/api_policy.py
src/tinyops/policies/fewshot.py
src/tinyops/eval/run_eval.py          # extend from skill 4
configs/eval.yaml
notebooks/01_baselines.ipynb
tests/test_parsing.py tests/test_run_eval.py
```

## 5.1 Parser — `policies/parsing.py`
`parse_tool_call(text: str) -> ParseResult(ok: bool, call: ToolCall | None, format: str | None, error: str | None, n_calls_found: int)`
Accepted formats, tried in order:
1. **AgentTune fallback list format** (verbatim from api_notes), e.g. `[{"tool_name": "describe_pod", "parameters": {...}}]` → `format="fallback_list"`.
2. Cohere action block `<|START_ACTION|>[...]<|END_ACTION|>` with `tool_name`/`parameters` → `format="cohere_action"`.
3. A single JSON object `{"tool_name"|"name"|"tool": ..., "parameters"|"arguments"|"args": {...}}` → `format="json_object"`.
4. Any of the above inside markdown fences ``` ```json … ``` ``` → same format with `_fenced` suffix.
Rules:
- Extract the **first** balanced JSON array/object (write a small bracket-matching scanner; don't rely on greedy regex). Tolerate leading/trailing prose.
- If more than one call is found, take the first and set `n_calls_found` (multi-call turns are recorded; the env executes one call per turn).
- Never silently fix malformed JSON (no trailing-comma repair, no quote fixing). Malformed → `ok=False` with error. The raw text is always logged.
- Normalize keys: `name`/`tool`→`tool_name`, `arguments`/`args`→`parameters`.

## 5.2 HF policy — `policies/hf_policy.py`
```python
class HFPolicy:
    def __init__(self, model_id: str, adapter: str | None = None, dtype: str = "auto",
                 load_in_4bit: bool = False, max_new_tokens: int = 256,
                 fewshot: list[Episode] | None = None, name: str | None = None): ...
    def act(self, messages) -> PolicyOutput
    def act_batch(self, batch_of_messages) -> list[PolicyOutput]
```
- dtype auto: `bfloat16` if `torch.cuda.is_bf16_supported()` else `float16`. `load_in_4bit` uses bitsandbytes NF4 (eval only; record it in episode meta).
- Adapter via `peft.PeftModel.from_pretrained(base, adapter)`; keep unmerged for eval.
- Render with `tokenizer.apply_chat_template(messages, add_generation_prompt=True)`. Tiny Aya's template has no tool role, so tool results are user turns in AgentTune's `tool_result_format` — exactly as built in skill 2.
- Greedy decoding (`do_sample=False`), `max_new_tokens=256`. Stopping: a `StoppingCriteria` that stops once a complete balanced JSON array/object has been emitted after the first `[` or `{` (plus EOS).
- `act_batch`: left padding, pad token = eos if missing; batch size from config (T4: 8, L4: 16). Different episodes progress in lockstep; finished episodes drop out of the batch.
- Return usage: prompt/completion token counts and wall-clock latency per turn.

## 5.3 API policy — `policies/api_policy.py`
`APIPolicy(model="gemini/gemini-2.5-flash", temperature=0, max_tokens=256)` via `litellm.completion`; same messages; retries with exponential backoff (3 tries); records provider-reported token usage. Rate limit with a simple semaphore (configurable RPM). Look up the current model name on the provider docs on the day and record it in DECISIONS.md.

## 5.4 Few-shot — `policies/fewshot.py`
`build_fewshot_messages(k=2, seed=1234)` picks k oracle episodes **from the train split** (different failure types from each other) and renders them as prior conversation turns after the system prompt. Must never use val/test scenarios.

## 5.5 Eval runner — `eval/run_eval.py`
CLI:
```bash
python -m tinyops.eval.run_eval --policy hf --model CohereLabs/tiny-aya-global [--adapter <path|hub id>] \
   [--fewshot 2] --split val --limit 0 --batch-size 8 --seed 1234 \
   --out results/episodes/tinyaya_zeroshot_val.jsonl
python -m tinyops.eval.run_eval --policy api --model gemini/gemini-2.5-flash --split test ...
```
- Resumable: if `--out` exists, skip scenario ids already present; append each finished episode immediately (`append_jsonl`). Checkpoint is therefore per episode — safe against Colab disconnects.
- Writes a run manifest `results/runs/<run>.json`: policy config, model id, adapter, dtype/quantization, prompt version, git commit, env.json, split, seed, start/end time, GPU hours used (append to `results/compute_log.csv`: date, run, gpu, hours).
- At the end, calls `score_episodes` to produce `results/metrics/<run>.json|.md`.
- `--split test*` with a non-final model prints a red warning; skill 5 runs base models only, so test evaluation of **base** models is allowed (they're not being tuned on it), but all **choices** must be made from val.

## 5.6 Notebook — `notebooks/01_baselines.ipynb`
Cells:
1. GPU check (`nvidia-smi`), Python version; create a 3.12 venv if needed.
2. `git clone` the repo, `pip install -e '.[train]'`, HF login from Colab secrets, `python -m tinyops.utils.envinfo`.
3. Fetch data (from git or HF dataset repo) and **verify `TEST_HASHES.txt`**.
4. Tiny Aya zero-shot on `val` (180).
5. Tiny Aya few-shot (k=2) on `val`.
6. Tiny Aya zero-shot on `test`, `test_injection`, `test_ood` (the "before" numbers).
7. Optional: Aya Expanse 8B zero-shot on `val` (4-bit on T4 if needed).
8. Push `results/` episodes + metrics to the HF dataset repo (`<user>/tinyops-data`, folder `baselines/`).
Gemini runs are done from the laptop (CPU, API) on `test` and `test_injection`.

## 5.7 Failure-mode analysis (write into `docs/BASELINE_FINDINGS.md`)
From val episodes:
- Table of TinyScore headline metrics for every baseline (with CIs).
- Histogram/table of `ParseResult.error` categories and parse formats used.
- Top 5 failure modes with counts and 1 short example each (raw model text): e.g. prose instead of tool call, invented tools, wrong argument names (`pod_name` vs `pod`), repeating `list_pods`, never submitting, obeying injected instructions, submitting without evidence.
- What each failure mode implies for SFT data and GRPO rewards (feeds skills 6–7).

## Tests (CPU)
- `test_parsing.py`: ≥ 20 cases — each accepted format; fenced; prose before/after; two calls; nested braces inside strings; malformed JSON (must fail); empty text; key aliases; non-dict parameters (fail).
- `test_run_eval.py`: run with a fake policy (a stub returning canned texts) on 5 val scenarios; resume skips done ids; manifest written; metrics produced.

## Exit gate
- [ ] Baseline metrics files exist for Tiny Aya zero-shot + few-shot (val, test, test_injection, test_ood) and Gemini (test, test_injection).
- [ ] `docs/BASELINE_FINDINGS.md` lists the top failure modes with real examples.
- [ ] Raw episodes uploaded to the HF dataset repo.
- [ ] `pytest -q` green. Commit `skill-5: policies, parser, eval runner, baselines`.
