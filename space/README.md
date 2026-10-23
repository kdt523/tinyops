---
title: TinyOps demo
emoji: 🛟
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
license: cc-by-nc-4.0
short_description: Tiny Aya investigates a simulated Kubernetes incident and ignores injected instructions
models:
  - <user>/tinyops-tiny-aya
  - <user>/tinyops-tiny-aya-sft
datasets:
  - <user>/tinyops-data
---
# TinyOps demo

Pick a simulated Kubernetes incident, choose the base Tiny Aya or the TinyOps model, and watch the tool-call
transcript. Each step is badged ✅ valid, ⚠️ schema error / unknown tool / unparseable, or ❌ unsafe. For
incidents with an injected instruction the payload is shown above the transcript.

- **Replay mode** (default, always works) plays pre-recorded episodes of the base and the final model on the
  *same* incident from the locked test splits.
- **Live mode** appears only when the Space has a GPU and the model stack; it runs KubeSim and the model step by step.

**How the replays were chosen (no highlight reel).** Fixed, seeded quotas over four outcome buckets: base fails and
final succeeds (8), both succeed (4), both fail (4), base succeeds and final fails (4); quotas move to the next
bucket if one runs short. The realised counts are in `replays/index.json`. The table below the demo is copied from
`results/final/report.json`.

Links: model `<user>/tinyops-tiny-aya`, dataset `<user>/tinyops-data`, article and code repository: see the model card.
Research use only: Tiny Aya is CC-BY-NC; AgentTune (LSAL v1.1) and AuditKIT (LSAL v1.2) are source-available.
