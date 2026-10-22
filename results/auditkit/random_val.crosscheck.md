### TinyScore vs AuditKIT: random_val on val (n=180)

| Metric | TinyScore | AuditKIT | Diff (pts) | Flag |
|---|---|---|---|---|
| Diagnosis success vs AuditKIT verified outcome | 0.0% | 0.0% | +0.0 | ok |
| Valid-call rate vs tool_call_validity | 100.0% | 100.0% | +0.0 | ok |
| Redundant-call rate vs redundant_tool_calls | 6.2% | 6.2% | +0.0 | ok |
| Loop-free rate (not max_steps) vs agent_loop_detection | 100.0% | 87.8% | +12.2 | DISAGREE |
| Permitted-call rate (no delete_namespace) vs tool_permission | 99.1% | 99.1% | +0.0 | ok |
