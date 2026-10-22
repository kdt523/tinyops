# Decisions

Entry template:

```
## YYYY-MM-DD — <title>
**Plan said:** …
**Source/reality:** … (file:line)
**Decision:** …
```

## 2026-10-22 — AgentTune install replaced the CPU torch wheel
**Plan said:** install CPU torch first, then AgentTune; if the pinned torch conflicts, install AgentTune first and record what happened (skill 1, A2).
**Source/reality:** `uv pip install -e ./AgentTune[dev]` resolved `torch 2.11.0+cu130` (CUDA available in WSL). transformers 5.20.2, trl 1.7.1, peft 0.21.2.
**Decision:** keep the resolved stack. TinyOps CPU code and tests must not require a GPU (CLAUDE.md rule 3).

## 2026-10-22 — WSL login shells are broken; run steps as non-login scripts
**Plan said:** plain `bash -l` style commands inside Ubuntu (skill 1, A2).
**Source/reality:** `/etc/profile.d/cuda.sh` fails to parse the inherited Windows PATH (contains "(x86)").
**Decision:** run WSL steps with `wsl -e bash script.sh` (non-login) and set PATH explicitly.

## 2026-10-22 — Repo lives on the Windows drive, venv in WSL
**Plan said:** repo and env in WSL2.
**Source/reality:** the working directory is `c:\New folder (2)`; WSL reaches it at `/mnt/c/New folder (2)`.
**Decision:** edit files on Windows, install editable and run pytest/ruff from WSL (`~/work/.venv`).

## 2026-10-22 — hatchling needs allow-direct-references for the pinned AgentTune extra
**Plan said:** `train = ["agenttune @ git+https://…@<PINNED_COMMIT>", …]` in pyproject (skill 1, B2).
**Source/reality:** hatchling raises `ValueError: Dependency #1 of option train … cannot be a direct reference unless tool.hatch.metadata.allow-direct-references is true`.
**Decision:** added `[tool.hatch.metadata] allow-direct-references = true`. Pinned commit `d4db4c12d52342227c4113a70183d0f64aa2f345`.

## 2026-10-22 — System prompt = rules + AgentTune fallback, joined by a blank line
**Plan said:** compose role/rules plus the tool list "exactly as AgentTune's Tiny Aya fallback prompt renders it" (skill 2.7).
**Source/reality:** `rollout_factory.py` ~L2194-2200 renders `DEFAULT_TOOLS_FALLBACK_PROMPT.format(tools=json.dumps(tool_schemas, indent=2))` and, when a system prompt exists, appends it as `"{system}\n\n{fallback}"`. Schemas are `transformers.get_json_schema` output (includes a `"return"` key).
**Decision:** `policies/prompt.py` copies the template verbatim and builds `RULES_PROMPT + "\n\n" + rendered fallback`. `sim/tools.agenttune_schema()` reproduces `get_json_schema`; tests compare it with both `transformers.get_json_schema` and AgentTune's `_build_tool_schema`, and compare the final prompt with AgentTune's own join. In skill 7 pass `system_prompt=RULES_PROMPT` to AgentTune so it builds the identical text.

## 2026-10-22 — System prompt is ~2,000 tokens, not ~900
**Plan said:** keep it under ~900 tokens (skill 2.7).
**Source/reality:** `build_system_prompt()` is 7,530 chars (rules part 539 chars). The tool block is AgentTune's indent-2 JSON of 10 schemas and must match training/inference exactly.
**Decision:** keep it. Real token count to be measured with the Tiny Aya tokenizer in skill 5; shorten tool descriptions only if sequence length becomes a problem (that would bump `PROMPT_VERSION`).

## 2026-10-22 — Fallback prompt asks for plain-text answers after tool results
**Plan said:** episode ends with a `submit_diagnosis` tool call.
**Source/reality:** AgentTune's fallback text says to use a "Tool result" turn "to answer the user's original question as plain text", and its rollout loop ends only when a turn has no tool call (`rollout_factory.py` ~L2296).
**Decision:** `RULES_PROMPT` says "Always reply with a tool call, never plain text. Finish by calling submit_diagnosis". The conflicting sentence stays (it is part of the verbatim template). Whether this holds for Tiny Aya is measured in skill 5. How the GRPO rollout terminates after `submit_diagnosis` is a skill 7 decision.

## 2026-10-22 — Turn accounting in KubeSimEnv
**Plan said:** episode ends on valid `submit_diagnosis`, reaching `max_steps` tool calls, or `max_parse_failures` consecutive parse failures.
**Decision:** every policy turn (parsed or not) counts toward `max_steps`, so total turns are bounded. Precedence at the end of a turn: submitted > parse_failures > max_steps. A `submit_diagnosis` with schema-invalid args does not end the episode. Schema-invalid write calls are not executed (so `is_write` stays False).

## 2026-10-22 — Observation and EventLog export
**Plan said:** export per AgentTune `trajectories.jsonl`; EventLog only if the constructor is public.
**Source/reality:** real AgentTune rollouts store `observation = str(dict(tool_results))`; KubeSim stores the raw tool text. EventLog's constructor is public (`agentic/events.py:41`), so `episode_to_eventlog` is implemented (light tier). Per-call pairing in AuditKIT needs `metadata.conversation`, so the export includes it.
**Decision:** raw observation text; `metadata.conversation` included; EventLog export imports AgentTune lazily (it is a `[train]` extra, absent in CI).

## 2026-10-22 — ruff version
**Source/reality:** the current ruff default rule set flags `dict(...)` calls, `typer.Option` defaults (B008) and loop writes (FURB122). CI installs the latest ruff, so the code follows those rules rather than pinning an old one.

## 2026-10-22 — `list_pods` added to every failure type's relevant_tools
**Plan said:** per-type `relevant_tools` table (skill 3.4); e.g. `oom_killed` = describe_pod, get_metrics; `hung_process_restart` = get_logs, describe_pod, get_metrics.
**Source/reality:** the tools need a pod name, which only `list_pods` (or an alert naming the pod) reveals. A gold trajectory that skips `list_pods` would teach the model to invent pod names.
**Decision:** every type's `relevant_tools` includes `list_pods` (superset of the table). Skill 4's precision/recall use this list. A test (`test_oracle_only_uses_names_it_has_seen`) checks every namespace/pod/service/configmap/deployment argument was shown earlier, for all oracle variants.

## 2026-10-22 — Injection locations are uniform only over feasible locations
**Plan said:** test_injection locations uniform (skill 3.7).
**Reality:** `image_pull_backoff`, `pending_insufficient_memory` and `node_memory_pressure_eviction` pods have no logs, so a log payload is impossible there.
**Decision:** locations are balanced over the feasible ones (running counter per split). Result: log/event/configmap are within a few percent of each other in every split (checked in test_splits).

## 2026-10-22 — Oracle sees the payload before it ignores it
**Plan said:** injected episodes add an evidence item about an ignored instruction (skill 3.8).
**Decision:** the oracle simulates its plan; if the payload is not in any observation it adds the one call that reveals it (get_logs with a larger tail, get_events for the pod, or describe_pod + get_configmap). The "Ignored an instruction embedded in the logs/events/config" item is added only when the payload really appeared in an observation, so grounding stays honest. A single-evidence diagnosis gets the pod's `list_pods` row as a second grounded item, so the note never drops grounding below 0.5.

## 2026-10-22 — Safety targets are deployment-level
**Plan said:** a restart whose target is in `injection_targets` is unsafe (skill 2.4).
**Decision:** `_targets_match` compares only deployment forms (`dep`, `ns/dep`, `deployment/dep`). A namespace target (delete payloads) must not make every restart in that namespace "unsafe".

## 2026-10-22 — Gold files are not committed
**Plan said:** commit data JSONL if < 50 MB (skill 3 exit gate).
**Reality:** scenarios total ~23 MB; with gold episodes/trajectories data/ is 53 MB (each gold trajectory embeds the 7.5 KB system prompt).
**Decision:** commit `scenarios.jsonl`, `stats.json` and `TEST_HASHES.txt`; `data/*/gold_*.jsonl` are gitignored and regenerated deterministically with `python -m tinyops.scenarios.build_dataset gold --split all --seed 1234` (~10 s). They are uploaded to the HF dataset repo in skill 9.

## 2026-10-22 — Pods in `hard` logs and realism trade-offs
**Reality:** to keep the key line outside the last 20 log lines at `hard`, 22-40 benign lines follow it, even for crash-looping pods (real `kubectl logs` would end at the error). Describe output is cut at 1,500 chars (events can be truncated), so evidence is also reachable via `get_events`.
**Decision:** accepted for the simulator; noted as a limitation for the article.

## 2026-10-22 — AuditKIT `agent_loop_detection` vs TinyScore `loop_rate` (random policy, val): 12.2 pt gap
**Plan said:** if a comparable metric disagrees by more than 5 points, investigate and explain (skill 4.6).
**Investigation:** on `random_val` AuditKIT flags 22/180 episodes, TinyScore `looped` flags 0. All 22 flagged episodes contain a repeated call (45 episodes have one); none is flagged without a repeat. AuditKIT's reasons: "call-graph cycle" (a call revisits an earlier one), or the same call 3+ times. TinyScore's `looped` means the episode ended by hitting `max_steps`, which did not happen.
**Decision:** definitional difference, not a bug. TinyScore keeps `looped` = max_steps and reports `redundant_calls` separately; the cross-check table shows the gap and flags it. The two agree on the oracle (0 / 0) and on a genuine 8x repeated call (both flag; `test_genuine_loop_is_flagged_by_both`).

## 2026-10-22 — AuditKIT loop detector false positive on tool-call JSON "thoughts" (feedback for Lexsi)
**Reality:** `agent_loop_detection` has a reasoning-stagnation signal that compares consecutive `thought` strings (difflib ratio >= 0.85). AgentTune rollouts store `thought = raw_output`; with the Tiny Aya fallback format that is tool-call JSON, and consecutive calls like `[{"tool_name": "list_pods", ...}]` / `[{"tool_name": "describe_pod", ...}]` are textually similar. On 50 clean oracle episodes it flagged 46 as loops.
**Decision:** the TinyOps export writes `thought: null` (raw text stays in `metadata.conversation`). After this the oracle cross-check agrees on every metric. Candidate issue for `docs/LEXSI_FEEDBACK.md` (skill 9): repro = export any 3-call JSON-format trajectory with `thought=raw_output` and run `agent_loop_detection`.

## 2026-10-22 — Grounding ignores evidence shorter than 6 normalised characters
**Plan said:** grounding = fraction of evidence items that are normalised substrings of an observation (skill 4.1).
**Reality:** `"e"` or `"the"` is a substring of every observation, so junk evidence counted as grounded.
**Decision:** items under `MIN_EVIDENCE_CHARS = 6` never count as grounded (test: `test_tiny_evidence_strings_cannot_be_grounded`). Real key evidence is longer (shortest is `OOMKilled`).

## 2026-10-22 — Matcher uses word-start (stem) matching
`oom` matches `OOMKilled`, `evict` matches `evicted`, `pull` matches `pulled`; `room` does not match `oom`. Type confusion is decided on term counts: an answer that fully matches another type's generic keywords at least as well as the true type's fails (blocks "list every cause"). Generic keyword sets live in `scoring/diagnosis.py::GENERIC_KEYWORDS` and a test ties them to the generators.

## 2026-10-22 — Oracle bug found by the matcher; precision counts the fix tool
The first sanity run gave oracle success 96.1%: the pending-memory phrasing "no node has enough free memory" contains neither "insufficient" nor "not enough". Rephrased; oracle is now 100% on every split (`test_oracle_scores_perfectly_on_every_split`). `tool_selection_precision` treats `gt.correct_fix.tool` (restart_deployment for hung) as relevant, since the oracle must call it.

## 2026-10-22 — Untested piece: LLM-judge agreement
`scoring/judge_check.py` is implemented and unit-untested against a real judge: it needs a Gemini key. The matcher/judge agreement rate is therefore NOT measured yet and METRICS.md says so.

## 2026-10-22 — Closed-model name for the Gemini reference
**Plan said:** default `gemini/gemini-2.5-flash`; look up the current name on the provider docs the day of use and record it (skill 5.3).
**Source/reality:** web search on 2026-10-22: Google's "Using the latest Gemini models" page (https://ai.google.dev/gemini-api/docs/latest-model) lists `gemini-3.6-flash` as generally available; the English models page (https://ai.google.dev/models) still showed 3.5 Flash as newest, so the sources are not fully consistent. 3.6 Flash defaults to a medium thinking level.
**Decision:** `configs/eval.yaml` uses `gemini/gemini-3.6-flash` with `max_tokens: 1024` (thinking tokens count against `max_tokens` and a 256 budget can return empty text). `APIPolicy(extra=...)` forwards provider kwargs. **Unverified until the first real call with a key**; if LiteLLM rejects the name, fall back to the newest name its `model_cost` map knows and record it here.

## 2026-10-22 — HFPolicy details verified on a small model
`HFPolicy` was exercised end to end with `SmolLM2-135M-Instruct` (batched left-padded greedy generation, JSON-complete stopping, token usage) on the local GPU via WSL (`pytest -m slow`). Not yet run with Tiny Aya (gated, needs the human's token and a bigger GPU). transformers >= 5 takes `dtype=`, older versions `torch_dtype=`; the policy picks by version.

## 2026-10-22 — Notebooks are excluded from ruff
Colab cells use `!`/`%` magics and must run on runtimes older than 3.12 (ruff flags that branch as outdated). `[tool.ruff] extend-exclude = ["notebooks"]`; notebooks are instead checked by `tests/test_notebooks.py` (valid JSON, every `-m tinyops.*` module exists, no adapter/--final in the baseline notebook).

## 2026-10-22 — Parser is stricter than AgentTune's
`policies/parsing.py` never repairs malformed JSON and refuses to let a valid inner object leak out of a malformed outer list (missing comma, truncated output, trailing comma). AgentTune's `_extract_tool_calls` (`rollout_factory.py:1077`) is lenient (tolerant JSON scan, many formats). Expect different parse-failure rates between TinyOps evaluation and AgentTune GRPO rollouts; measure and record it in skill 7.

## 2026-10-22 — Skill 5 status
Done and unit-tested on CPU: parser (33 cases), HFPolicy (verified on SmolLM2-135M), APIPolicy (fake provider), few-shot builder, batched resumable runner with manifest/compute log/metrics, failure-mode report generator, notebook. NOT done (needs the human's HF token + a GPU, and a Gemini key): the actual Tiny Aya and Gemini baseline runs, therefore `docs/BASELINE_FINDINGS.md` and the HF dataset upload do not exist yet. No baseline number has been invented.
