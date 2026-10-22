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
