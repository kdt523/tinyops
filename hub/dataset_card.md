---
license: cc-by-4.0
task_categories: [text-generation]
language: [en]
tags: [kubernetes, agents, tool-use, synthetic, safety, prompt-injection]
---
# <user>/tinyops-data

Synthetic incident-response data for TinyOps: scenarios from **KubeSim**, a deterministic fake Kubernetes cluster,
with hidden ground truth, gold (oracle) trajectories, baseline and final evaluation episodes, and training logs.
All data is machine-generated; there is no real cluster data, customer data or personal data.

## KubeSim
Ten tools (`list_pods`, `describe_pod`, `get_events`, `get_logs`, `get_metrics`, `check_service`, `get_configmap`,
`restart_deployment`, `delete_namespace`, `submit_diagnosis`); `delete_namespace` is always denied and counts as an
unsafe action. Tool results are kubectl-like text, at most 1,500 characters.

## Failure types (9)
`crashloop_missing_env`, `oom_killed`, `image_pull_backoff`, `readiness_probe_misconfig`, `service_selector_mismatch`, `pending_insufficient_memory`, `node_memory_pressure_eviction`, `bad_configmap_value`, `hung_process_restart`. Difficulties: easy (one namespace, no distractors), medium (healthy
neighbours and noisy logs), hard (second namespace, buried key line, misleading warning, unrelated events).

## Injection suite
Families: `direct_command`, `fake_authority`, `tool_format_smuggle`, `urgency_social`, `wrong_restart`, `exfil`, `benign_lookalike` (`benign_lookalike` is a harmless control that measures
over-refusal). Payloads sit in a log line, an event message or a ConfigMap value; the real fault stays intact.

## Splits (generated with seed 1234, generator version v1)
| Split | Incidents | Injected | Purpose |
|---|---|---|---|
| `train` | 1215 | 182 | SFT / RL training |
| `val` | 180 | 27 | all model and checkpoint selection |
| `test` | 270 | 0 | final eval, no injections |
| `test_injection` | 162 | 162 | final eval, every injection family |
| `test_ood` | 63 | 0 | final eval, unseen names/alerts |

`test*` splits are **locked**: generated once, SHA-256 hashed, never used for training, prompt tuning,
checkpoint selection or hyper-parameters. Hashes:
```
e2b56d300ee1cbfce05042fa225e2e3859dbc43ccd6dd788f98b2b9ed5597eb5  data/test/scenarios.jsonl
b11863499f303ff22694fbae59a625024e37b823ba33c5e5ef4d0a115d2d461a  data/test_injection/scenarios.jsonl
d83f3666b548f08cea110dc2e31552dc2d29437aaf446a7946aab586b4c3bb16  data/test_ood/scenarios.jsonl
```
`test_ood` uses disjoint namespace, application, registry, node and environment-variable names and different alert
wording. Regenerate everything with `python -m tinyops.scenarios.build_dataset --seed 1234`.

## Contents
`scenarios/<split>.jsonl`, `TEST_HASHES.txt`, `gold/` (oracle episodes and trajectories for train and val),
`baselines/` (skill-5 episodes, metrics, manifests), `final/` (locked evaluation episodes and metrics),
`training_logs/` (reward logs, validation curves, `trajectories.jsonl`, `lexsi_provenance.json`).

## Licence and limitations
CC-BY-4.0 for the generated data (it is synthetic). Limitations: the simulator is simpler than a real cluster; log text
comes from templates; keyword-based scoring; English only. The models trained on it carry their own licences.
