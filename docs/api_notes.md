# API notes — AgentTune / AuditKIT / Tiny Aya (skill 1, Part A)

Verified 2026-10-22 in WSL2 Ubuntu-24.04, Python 3.12.3, venv `~/work/.venv`.
- AgentTune: `Lexsi-Labs/AgentTune` @ `d4db4c12d52342227c4113a70183d0f64aa2f345` (package version 1.1.3, editable install).
- AuditKIT: `auditkit==1.1.3` from PyPI.
- Resolved stack: torch 2.11.0+cu130, transformers 5.20.2, trl 1.7.1, peft 0.21.2.
- All AgentTune paths below are relative to `src/agenttune/`; AuditKIT paths relative to `site-packages/auditkit/`.

Status legend: **VERIFIED** = read in source and/or executed. **OPEN** = not yet verified (must be closed before the dependent skill).

## 0. Environment findings
- Installing `AgentTune[dev]` replaced the CPU torch wheel with `torch 2.11.0+cu130` (`cuda.is_available() == True` in WSL). CPU-only torch was not preserved; CLAUDE.md rule 3 (CPU path always works) is unaffected because all TinyOps CPU code must not require a GPU.
- Repo layout differs from the skill text: AgentTune is a `src/` layout, docs are in `docs/` (`docs/community/known-issues.md`, `changelog.md`).
- WSL login shells fail (`/etc/profile.d/cuda.sh` chokes on the inherited Windows PATH containing "(x86)"). Run WSL steps as `bash script.sh` (non-login).
- **OPEN:** `.wslconfig` (memory=8GB, swap=8GB) not created yet; WSL currently sees ~7 GB RAM.

## 1. AgentTune — trainer factory (VERIFIED)
`core/backend_factory.py:130`
```python
def create_agentic_trainer(algorithm: str, backend: str = "auto", **kwargs) -> Any
```
- `algorithm` in {grpo, dpo, ppo, rloo, bco}; `backend` in {auto, trl, unsloth}. `auto` picks Unsloth if importable else TRL — **pass `backend="trl"` explicitly**.
- All `**kwargs` are forwarded verbatim to the trainer class (`TrlAgenticGrpo`, etc.).
- If `reward_funcs` is more than a single callable (or `reward_weights` given), the factory wraps it with `combine_rewards` (`backend_factory.py` ~L200).
- GRPO class: `backends/trl/agentic/grpo/agentic_grpo.py`. `_split_kwargs` (L78) routes each kwarg by introspecting `GRPOConfig.__init__` and `GRPOTrainer.__init__`; unknown keys are **silently discarded**. A typo'd kwarg therefore does nothing — always log the effective config.
- Defaults set in `setup_trainer` (L375+): `num_generations=4`, `max_completion_length=256`, `temperature=0.7`, `top_p=0.95`, `beta=0.04`, `learning_rate=1e-6`, `gradient_accumulation_steps=16`, `save_steps=100`, `warmup_steps=10`, `remove_unused_columns=False`.
- `peft_config`: dict -> `peft.LoraConfig(**dict)`, or a `PeftConfig` object (`_resolve_peft_config`, L109). No path to load an existing adapter as trainable. **OPEN for skill 7:** whether passing an already-`PeftModel` as `model=` works; safe default is to merge the SFT adapter into the base first.
- `max_steps_per_turn` (default 20 in GRPO, L459) = cap on the tool loop per rollout (passed as `max_steps` to the rollout fn). Note: TRL's own `max_steps` is the optimizer step count — different thing.
- `trajectories_file` (default `"trajectories.jsonl"`, `None` disables) -> `<output_dir>/trajectories.jsonl` (L454).
- `tools_fallback_prompt`, `tool_result_format` are forwarded to `create_rollout_fn` only if present in kwargs (L468).
- `tools`: python callables or `BaseTool` objects. Schema built by `agentic/rollout_engines/rollout_factory.py:706 _build_tool_schema` -> `transformers.utils.get_json_schema(fn)` (needs type hints + Google-style `Args:` docstring); on any exception falls back to `_fallback_tool_schema` (L675) which types **every** param as `"string"`. A test in skill 2 must compare against `TOOL_SCHEMAS` to catch silent fallback.

## 2. AgentTune — rollout loop (VERIFIED, source only)
`agentic/rollout_engines/rollout_factory.py`
- **Tool-call parsing** `_extract_tool_calls` (L1077): strips `<think>…</think>`; native `tool_calls` first; then Cohere parsers (`_parse_cohere_action` = `<|START_ACTION|>[…]<|END_ACTION|>`, `_parse_cohere_legacy_action`, `_parse_cohere_json`) — this is where the fallback list is parsed; then generic `<tool_call>` JSON; then Llama formats. It is **lenient** (accepts many formats). TinyOps' own parser (skill 5) must be strict; the two will disagree on edge cases — record in DECISIONS when found.
- **Loop** (L2296): `while tool_calls and iteration_num < max_steps`. Tool execution `_run_tools_parallel` (L767). **A turn with no parsed tool call ends the rollout** (it is treated as the final answer). Unknown tool -> `{"error": "Tool 'x' not found"}` observation, counted in `tool_failure_count`.
  - Consequence for skill 7: `submit_diagnosis` is just another tool; the rollout does not stop on it. The model must emit a non-tool turn afterwards, or `max_steps_per_turn` must be reached. **OPEN:** design how TinyOps ends an episode under AgentTune's loop (e.g. `submit_diagnosis` result tells the model to reply with plain text "Done", and rewards read the diagnosis from the recorded call). Must be resolved in skill 7 (7.1).
- **Parse failure** is not an error event: no call parsed => loop ends. There is no "parse-failure retry" like KubeSim's `max_parse_failures`. So training episodes and eval episodes terminate differently on malformed output; document in METRICS when building rewards.
- **Tool results** are appended as `role: "tool"` messages, then (for Tiny Aya, whose template ignores `tools`) `fold_tool_messages_into_user` (L108) turns each into a user turn using `tool_result_format`.
- **Tools receive only the model's arguments** (`fn(**arguments)`, `_run_sync_or_async` ~L735). **No dataset column reaches tools.** => skill 7 must use the ContextVar/registry approach (7.2 "otherwise" branch). `_run_tools_parallel` runs calls in threads/asyncio, so a `ContextVar` set in the caller thread may NOT propagate — **OPEN: test isolation explicitly.**
- **Rewards** receive: `reward_fn(completions, prompts=prompt_texts, **extra)` where `extra` = every other dataset column as a same-named kwarg + `tool_call_counts` (+ `answer` when set); falls back through 3 older call styles on `TypeError` (L1388–1400). For the GRPO path the rollout returns `"trajectories"` as an extra field (L1449), which TRL forwards to reward functions as kwarg `trajectories` (list of `Trajectory`) — so rewards **can** see the full multi-turn trajectory (steps with action/observation). **OPEN:** confirm by a CPU smoke run in skill 7.
- **num_generations execution (sequential vs batched):** **OPEN** (not read). Matters for per-rollout state.

## 3. AgentTune — exact Tiny Aya fallback prompt (VERIFIED, VERBATIM)
`rollout_factory.py:537`
```python
DEFAULT_TOOLS_FALLBACK_PROMPT = (
    "You can call these tools:\n{tools}\n\n"
    "To call tools, reply with only a JSON list, one object per call, and no other text: "
    '[{{"tool_name": "<name>", "parameters": {{"<argument>": <value>}}}}]. '
    "Use a list even for one call. "
    'A user turn that starts with "Tool result" is the output of your call, not a new '
    "message from the user: use it to answer the user's original question as plain text."
)
```
Rendered (L2194): `tools_fallback_prompt.format(tools=json.dumps(tool_schemas, indent=2))` — `{tools}` is the JSON dump of the **schema list** returned by `_build_tool_schema` (OpenAI-style `{"type":"function","function":{name,description,parameters}}`), indent=2. Falls back to `.replace("{tools}", …)` if `.format` fails.

`rollout_factory.py:102`
```python
DEFAULT_TOOL_RESULT_FORMAT = (
    "Tool result for {name}: {content}\nUse this result to answer the original question."
)
```
- **Output format the model must produce:** `[{"tool_name": "<name>", "parameters": {...}}]` — a JSON **list**, one object per call.
- The fallback text is injected as a **system prompt**; TinyOps' own system prompt (skill 2.7) must be composed so that the final text AgentTune would produce and the text TinyOps uses at SFT/eval time are identical. **OPEN:** exactly how `system_prompt=` and the fallback are combined (concatenated? replaced?) — read `rollout_factory.py` ~L2181–2200 in skill 2 before writing `policies/prompt.py`.
- Note the instruction "use it to answer the original question as plain text" conflicts with TinyOps' `submit_diagnosis` terminal tool. Skill 2/7 must decide how the system prompt overrides this; log in DECISIONS.md.

## 4. AgentTune — rewards (VERIFIED)
- `agentic/rewards/builtin_rewards/__init__.py:17 REWARD_REGISTRY` — 26 SQL/use-case/FinQA/PRM rewards; none are TinyOps-relevant. We write our own (skill 7). Known-bad per CLAUDE.md: `distilled_judge_reward`, Rouge/Bleu.
- `agentic/rewards/composite.py:33 combine_rewards(reward_funcs, weights=None)`: accepts callables or registry names; **weights are normalised to sum to 1** (so `weights={1.0,0.5,…}` are relative, not absolute); calls each as `fn(completions=…, **kwargs)`, retries positionally on `TypeError`; `None` score -> 0.0. Returns a single callable; **individual components are not logged separately** — TinyOps must log its own breakdown inside reward fns.
- Signature to implement: `def r_xxx(completions, **kwargs) -> list[float]` (keyword-capable; `prompts` and dataset columns arrive as kwargs).

## 5. AgentTune — distillation / SFT (VERIFIED)
`agentic/distill_trainer.py:277 create_distill_trainer(student, *, demonstrations=None, trajectories=None, teacher_engine=None, tasks=None, rollout_kwargs=None, tools=None, reward_fn=None, system_prompt=None, max_steps_per_turn=8, fmt="sft", trainer_factory=None, use_peft=True, lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_bias="none", lora_targets=None, task_type="CAUSAL_LM", dtype=None, device_map="auto", probe_prompt=None, trust_remote_code=False, output_dir="./out_distill", **sft_kwargs) -> DistillTrainer`
- Inputs: `demonstrations` = (prompt, response) pairs or dicts; `trajectories` = list of `EventLog` (full tier); or `teacher_engine`+`tasks`.
- **Training row format differs from Tiny Aya's fallback format:** `agentic/events.py` `as_dataset_rows("sft")` renders tool calls as `<tool_call>{"name":…,"arguments":…}</tool_call>` assistant messages (not the `[{"tool_name":…,"parameters":…}]` list). Training on that would teach a format the Tiny Aya fallback prompt does not ask for. **Decision for skill 6 (to be logged then): use the plan's fallback path — TRL `SFTTrainer` + `LoraConfig` on our own dataset.**
- Only `fmt="sft"` supported. Needs full-tier `EventLog`s; building them by hand needs token spans (`as_dataset_rows` calls `_require_full`).

## 6. AgentTune — EventLog (VERIFIED)
`agentic/events.py:41`
```python
@dataclass class EventLog: events: list[Event]; tier: str = "light"; id: str = uuid4
@dataclass class Event: kind: EventKind; payload: dict; token_span: tuple[int,int]|None; logprobs: list[float]|None; scope: str|None
EventKind: TEXT, REASONING, TOOL_CALL, TOOL_RESULT, OBSERVATION, TURN_COMPLETE, REWARD, MEMORY_OP
```
Public constructor; a light-tier log can be built by hand (`EventLog(events=[Event(EventKind.TOOL_CALL, {...}), …], tier="light")`). Full-tier-only operations (`masked_tokens`, `as_dataset_rows`) raise on light logs. Reward events carry `payload["value"]` and `scope` in {step, turn, episode}. `episode_from_eventlog` in AuditKIT accepts these (see 8).

## 7. AgentTune — `trajectories.jsonl` line format (VERIFIED)
`agentic/trajectory/dataset.py:44 Trajectory`, `:53 to_dict`, `:156 trajectory_writer`
```
{"task": str, "steps": [{"step_number": int, "state": str,
   "action": {"tool_calls": [{"id": str|null, "name": str, "arguments": dict}]},
   "observation": str, "thought": str|null, "reward": float|null}],
 "reward": null, "trajectory_id": str, "final_response": str, "metadata": {...}}
```
- `observation` in real rollouts is `str(dict(tool_results))` (a Python-repr dict keyed by tool name), not the raw tool text. AuditKIT reads it as the tool output; KubeSim export (skill 2) should store the raw observation string to keep grounding checks simple, and note the difference.
- `training-only` metadata keys are stripped in `to_dict`; `reward` is written as null unless `keep_reward=True`.

## 8. AuditKIT (VERIFIED)
- Quickstart executed: `ak.evaluate(dataset, model, scorers=[...])` — **note argument order is (dataset, model, scorers)**. `ak.evaluate([ak.Sample(input="2+2?", target="4"), …], lambda prompts: [...], scorers=["exact_match"])` returned `RunResult`; `.summary()` printed `exact_match: 0.5000 ± 0.3536 (n=2)`.
- `ak.Sample(input, target=None, id=None, task="", kind=…, choices=None, retrieval_context=None, metadata, tags, actual_output=None, tools=None, expected_tool_calls=None, reference_contexts=None, actual_trace=None, images=None)`.
- `agent_eval/importers.py:515 episodes_from_agenttune(path) -> list[AgentEpisode]` — executed on a hand-built one-line trajectories.jsonl: produced 1 `AgentEpisode` with `AgentEvent(type="tool_call"|"tool_result", payload={"id","name","arguments"}|{"output"})`. `:572 episode_from_eventlog(log, *, source_id=None)`; `:280 _load_agenttune_pairs`; `:681 inspect_agenttune(path)` for format inspection.
- Outcome verifier hook: `agent_eval/outcome.py:192 CustomPredicate(fn, *, version, name=None)` — `fn(episode, case) -> bool | "success"|"failure"|"unknown"|"error"`; **`version=` is mandatory**. Also `AnswerAssertion`, `FinalStateAssertion`, `ArtifactAssertion`, `AssertionOracle`.
- `agent_eval/runner.py:256 case_from_episode(episode, *, outcome=None, case_id=None) -> AgentCase` (needs `case_id` or episode `case_id`/`source_id`; default outcome is an `AnswerAssertion` against `metadata["target"]`). `AgentEvalSpec` fields: `cases, mode, agent, agent_opts, episodes, import_path, trials, reset_confirmed, reliability_k, scorers, judge, name, config`; `AgentEvalRunner().run(spec) -> AgentEvalResult` (`.summary()`, `.to_json(path)`); modes `recorded|deployed|harness`.
- Agent scorers (`agent_eval/runner.py` `KNOWN_SCORERS`): `tool_call_f1, tool_call_f1_name, trajectory_match, parallel_tool_calls, tool_call_validity, redundant_tool_calls, retrieval, task_completion, agent_loop_detection, tool_permission, tool_selection`. Defaults: `tool_call_f1, tool_call_validity, task_completion`. **OPEN:** exact definitions of each (read `metrics/agent.py` in skill 4) before the cross-check table.
- RedTeam: `redteam/runner.py:110 RedTeamRunner(model)` where `model` is a `Model`, a **`Callable[[list[str]], list[str]]`**, or a string; `.run(probes=None, detectors=None) -> RedTeamResult` (`.summary()`). Probes (`redteam/probes/builtin.py`): `prompt_injection, jailbreak, encoding, refusal`. Detectors (`redteam/detectors/builtin.py`): `keyword, refusal, system_prompt_leak, injection_success`. Probes are single-turn prompts only.
- Significance: `model_compare.py:505 compare_models(models, dataset, scorers, …) -> CompareResult`; `.significance(a, b, metric)` uses a **paired bootstrap** by sample id (`_bootstrap.paired_bootstrap`), not McNemar. TinyOps keeps its own McNemar/paired-bootstrap (skill 4) and may cross-check.

## 9. Known issues relevant to us (docs/community/known-issues.md, read)
Matches CLAUDE.md rule 9: do not use `ToolRegistry.auto_register_builtins()`, no `--tools` CLI flag, `RougeMetric`/`BleuMetric`/`DistilledJudge` fail silently. CHANGELOG 1.1.0 documents the Tiny Aya fallback + `trajectories.jsonl` behaviour used above. CONTRIBUTING: Python >= 3.12; tests via `pytest tests/` (needs the `[dev]` extra because default addopts use `--cov`).

## 10. Tiny Aya on GPU (A5) — OPEN
Not run: requires a Colab/Kaggle GPU session with the human's HF token (licence gated). The local RTX 3050 has 4 GB VRAM — insufficient for fp16 3.35B. Pending items: generation smoke test, tokens/sec, VRAM, 2-step GRPO smoke run, load of its `trajectories.jsonl` in AuditKIT.

## 11. AgentTune test suite (A3.2) — OPEN
`pytest -q -x` run in WSL with a 1500 s `timeout`. It was killed by the timeout without a summary line: 56 tests passed (56 dots), no failure was reported before the kill. **Incomplete, not a pass.** To close: rerun in the background without `-x`, with `-n auto` (pytest-xdist is in the `[dev]` extra) and a longer timeout, then record passed/failed counts and any failing test names here.
