# Feedback for Lexsi Labs (AgentTune / AuditKIT)

Issues met while building TinyOps, with minimal repros. Versions: AgentTune 1.1.3 @ `d4db4c12d52342227c4113a70183d0f64aa2f345`,
AuditKIT 1.1.3, TRL 1.7.1, transformers 5.20.2, Python 3.12.3. Nothing has been filed yet — review, then open the issues
(security-sensitive ones via SECURITY.md instead). Tone: constructive; most of these are small.

## 1. AuditKIT `agent_loop_detection` flags clean tool-call trajectories as loops (false positive)
- **Repro:** import 3-call AgentTune trajectories whose `thought` is the raw model text, which with the Tiny Aya fallback
  format is JSON like `[{"tool_name": "list_pods", ...}]` then `[{"tool_name": "describe_pod", ...}]`; run the scorer `agent_loop_detection`.
- **Expected:** 1.0 (no repeated or cyclic call). **Actual:** 0.0 for 46 of 50 clean episodes; the "reasoning stagnation" signal
  (difflib ratio >= 0.85 between consecutive assistant/reasoning texts) fires on near-identical JSON.
- **Why it matters:** AgentTune's own rollouts store `thought = raw_output` (`rollout_factory.py` `Step(... thought=raw_output)`), so real GRPO
  trajectories in this format hit it. **Suggested fix:** skip the stagnation signal when the text parses as a tool call, or compare only free-text.
- **Our workaround:** export `thought: null` (see `docs/DECISIONS.md`, "AuditKIT loop detector false positive").

## 2. `create_agentic_trainer(...).trainer` is `None` until `.train()`
- **Repro:** `t = create_agentic_trainer("grpo", ...); t.trainer.add_callback(cb)` -> `AttributeError: 'NoneType' object has no attribute 'add_callback'`.
- **Expected/actual:** the docstrings and examples suggest a ready trainer; `setup_trainer()` runs inside `train()` (`agentic_grpo.py` `train()` L612).
- **Suggested fix:** document that callbacks go in as `callbacks=[...]` kwargs (this works), or build lazily with a property.

## 3. `create_distill_trainer` SFT rows use a different tool-call format from the Tiny Aya fallback
- `EventLog.as_dataset_rows("sft")` renders `<tool_call>{"name": ..., "arguments": ...}</tool_call>`, while for templates that ignore `tools`
  (Tiny Aya) the rollout asks for `[{"tool_name": ..., "parameters": ...}]` and parses that. Training on the former teaches a format
  the latter prompt does not request. **Suggested fix:** render distillation targets in the format the rollout will parse for the same tokenizer.

## 4. The fallback prompt asks for a plain-text answer after every tool result
- `DEFAULT_TOOLS_FALLBACK_PROMPT` ends with "...use it to answer the user's original question as plain text", while the rollout loop only ends on a turn with
  no tool call. With a terminal tool (`submit_diagnosis`) the two instructions conflict. **Suggested fix:** let the fallback text be overridden per task
  (`tools_fallback_prompt` already exists) or add an optional terminal-tool name that ends the loop.

## 5. Unknown tool names are answered by the loop, not by the tool layer
- `Tool 'x' not found` is returned as `{'error': ...}` before any registered callable runs, so environments cannot keep one consistent error text. Minor.

## 6. Installation notes
- `pip install -e ./AgentTune[dev]` replaces a pre-installed CPU torch wheel with a CUDA build (torch 2.11.0+cu130 here). Worth a note in the
  install docs for laptop users.
- transformers >= 5 renamed `warmup_ratio` (float `warmup_steps` is now the ratio); AgentTune's CLI/docs examples that pass `warmup_ratio` to TRL configs need a check.

## 7. AuditKIT: things that worked well (for balance)
`episodes_from_agenttune`, `CustomPredicate` outcome verification, `tool_call_validity` / `redundant_tool_calls` / `tool_permission` agreed with an independent
implementation to the decimal on 180 oracle episodes (`results/auditkit/oracle_val.crosscheck.md`).
