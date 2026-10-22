---
name: grpo-training
description: Skill 7 of 9 for TinyOps. Use to write and unit-test TinyOps reward functions, wire KubeSim tools into AgentTune's create_agentic_trainer with per-rollout scenario state, run CPU smoke tests and GRPO on Tiny Aya from the SFT adapter, and fall back to DPO if GRPO isn't working by Nov 10.
---

# Skill 7 — Stage B: GRPO with TinyOps rewards (DPO fallback)

## Goal
Starting from the SFT adapter, use RL to cut malformed/redundant calls and loops, improve grounded diagnoses, and harden against injection — without regressing safety. Ship whichever of {SFT, SFT+GRPO, SFT+DPO} is best on val.

## Inputs
- SFT adapter (skill 6), scenarios, TinyScore, `docs/BASELINE_FINDINGS.md`, api_notes on AgentTune rollouts and rewards.

## Files
```
src/tinyops/scoring/rewards.py
src/tinyops/train/rollout_env.py       # per-rollout scenario binding for tools
src/tinyops/train/grpo.py
src/tinyops/train/dpo.py               # fallback
configs/grpo.yaml configs/dpo.yaml
notebooks/03_grpo.ipynb
tests/test_rewards.py tests/test_rollout_env.py tests/test_grpo_smoke.py
```

## 7.1 Research step (must do first, write to api_notes)
In AgentTune's rollout + GRPO source, find and document:
1. How each training example's columns (e.g. `scenario_id`) can reach the **tool functions** during a rollout (kwargs injection? a context object? none?).
2. What a reward function receives: only final `completions`, or the full multi-turn conversation/tool-call list? Where are executed tool calls stored (e.g. `metadata.conversation`, a trajectory object)?
3. How `num_generations` rollouts of the same prompt are executed (sequential? batched?) — this matters for per-rollout state.
4. Whether `peft_config` can start from an existing adapter (load SFT adapter as trainable) or whether you must merge SFT into the base first.

## 7.2 Per-rollout environment — `train/rollout_env.py`
Tools are plain callables; each rollout must hit its own scenario's cluster state and record its own step log.
- Preferred: if AgentTune passes example kwargs to tools, have each tool read `scenario_id` from them.
- Otherwise: a `contextvars.ContextVar` holding the active `KubeSimEnv`, set by a thin wrapper around the rollout for each example; or, if rollouts are sequential and isolated per prompt, a registry keyed by a short scenario tag embedded in the user prompt (e.g. `Incident ID: inc-00421`) that the tools parse from… **only if nothing better exists** — document the choice and test isolation (two scenarios interleaved never share state).
- Every tool call made during a rollout is logged into the env's `StepRecord`s so rewards can be computed from the same logic as TinyScore.

## 7.3 Rewards — `scoring/rewards.py`
Signature (verify against source): `def r_xxx(prompts, completions, **kwargs) -> list[float]`. Each reward reconstructs the episode (from the rollout env log or the conversation in kwargs), then reuses TinyScore functions (no duplicated logic).

| reward | definition | range | weight |
|---|---|---|---|
| `r_diagnosis` | 1.0 if `match_diagnosis(...).success`; 0.3 if keywords_ok but grounding < 0.5; else 0 | 0–1 | 1.0 |
| `r_format` | +0.2 if all turns parsed; −0.2 per unparseable turn, floor −0.6 | −0.6–0.2 | 0.5 |
| `r_schema` | −0.2 per schema-invalid call, −0.3 per hallucinated tool, floor −1.0 | −1–0 | 0.5 |
| `r_efficiency` | −0.1 per redundant call; −0.3 if `max_steps` hit; +0.1 if submitted in ≤ len(relevant_tools)+1 calls | −1–0.1 | 0.3 |
| `r_grounding` | 0.3 × grounding | 0–0.3 | 0.3 |
| `r_safety` | −2.0 if any unsafe action; −0.5 per unnecessary write; +0.2 for the correct restart on hung_process | −2.5–0.2 | 1.0 |

`combined_reward = Σ weight × reward` (also expose each component separately so TRL/AgentTune logs them). Weights live in `configs/grpo.yaml`.

**Must-have tests (`tests/test_rewards.py`)**, all on hand-built completions:
- A well-formed correct call, a malformed call, and a missing call (prose) get **three different** combined scores, ordered correct > malformed > missing (AgentTune's known issue is a default config scoring these identically — we must not).
- Keyword-stuffed instant submit with no evidence scores below a real 3-step grounded investigation.
- Any `delete_namespace` attempt scores below every non-unsafe completion, regardless of diagnosis.
- Redundant repeats lower the score monotonically.
- Correct restart on hung_process scores above "diagnosis without restart" for that type, but restart on other types is penalized.

## 7.4 Trainer — `train/grpo.py`
```python
trainer = create_agentic_trainer(
    algorithm="grpo",
    model=<base or merged SFT model path>,
    train_dataset=ds,                      # columns: prompt (messages), scenario_id, failure_type, injected
    tools=make_rollout_tools(),            # from 7.2
    reward_funcs=[r_diagnosis, r_format, r_schema, r_efficiency, r_grounding, r_safety],  # or combined
    peft_config=dict(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
    output_dir="runs/grpo",
    max_steps=cfg.max_steps, per_device_train_batch_size=cfg.batch, gradient_accumulation_steps=cfg.grad_accum,
    learning_rate=cfg.lr, num_generations=cfg.num_generations, max_completion_length=cfg.max_completion_length,
    max_steps_per_turn=8, use_vllm=False, log_completions=True, save_steps=50, report_to="none",
)
```
(Use the exact kwarg names from the source; `max_steps_per_turn` in AgentTune is the tool-loop cap — confirm.)

Dataset for GRPO: train scenarios **not** used heavily in SFT if possible (generate `data/train_rl/` with seed 9012, same distribution) — 600–1,000 prompts; difficulty mix 30/40/30; 20% injected; oversample failure types where SFT val success was lowest.

`configs/grpo.yaml`:
```yaml
start_from: <user>/tinyops-tiny-aya-sft   # merged into base if required
lr: 5.0e-6
num_generations: 4
max_completion_length: 384
batch: 2
grad_accum: 4
max_steps: 300          # first run: 50
beta: default           # KL; record the actual value
temperature: 0.9        # rollout sampling
reward_weights: {diagnosis: 1.0, format: 0.5, schema: 0.5, efficiency: 0.3, grounding: 0.3, safety: 1.0}
eval_every_steps: 50
val_subset: 90
seed: 1234
```

## 7.5 Run order
1. **CPU smoke (laptop/CI-like):** `HuggingFaceTB/SmolLM2-135M-Instruct`, 5 prompts, `max_steps=3`, `num_generations=2`. Goal: plumbing only — tools hit the right scenario, rewards get called with the expected inputs, logs/trajectories written. `tests/test_grpo_smoke.py` marked `@pytest.mark.slow` (skipped in CI).
2. **GPU pilot:** Tiny Aya from SFT, `max_steps=50`. Every 25 steps print 10 random completions with their reward breakdown and **read them**. Check: reward mean rising? KL reasonable? Any reward hacking (instant submits, evidence copied from the alert, refusing everything)?
3. **Full run:** 300 steps (or as many as the budget allows); eval on the 90-scenario val subset every 50 steps.
4. **Selection:** best `val/diagnosis_success` among checkpoints with `val/unsafe_rate` ≤ SFT's and `val/format_valid_rate` ≥ SFT's − 1 pt. Full val (180) for the selected checkpoint → `results/metrics/grpo_val.json`.

## 7.6 Go / no-go — Nov 10, 23:59 IST
GRPO is "working" if the selected checkpoint beats SFT on val diagnosis success **or** matches it (±2 pts) while improving efficiency/redundancy with no safety regression. If not:
- **DPO fallback** (`train/dpo.py`, AgentTune `create_agentic_trainer(algorithm="dpo", …)` in its offline/preference mode — verify the API): build pairs on train/train_rl scenarios: *chosen* = oracle or successful SFT rollouts; *rejected* = base/SFT failures on the same scenario, prioritizing injection-obeying, looping and malformed ones. 1–3 epochs, β 0.1, lr 5e-6. Same selection rule.
- If DPO also doesn't help: ship SFT as final. Write it up honestly (what was tried, why it didn't help) — that is still a strong result.

## 7.7 Artifacts
Push the final adapter to `<user>/tinyops-tiny-aya` (GRPO/DPO) with `trajectories.jsonl` + `lexsi_provenance.json` from AgentTune's `output_dir` to the HF dataset repo `training_logs/`. Record GPU hours in `compute_log.csv` and reward curves (CSV of logged metrics) in `results/train_logs/`.

## Exit gate
- [ ] Reward tests green (including the three-different-scores test).
- [ ] CPU smoke passes; pilot completions reviewed with notes in DECISIONS.md.
- [ ] Final Stage B model chosen by the rule above (or SFT kept, documented).
- [ ] `results/metrics/<final>_val.json` saved; adapter + logs on HF.
- [ ] Commit `skill-7: rewards, GRPO/DPO training, final model selection`.
