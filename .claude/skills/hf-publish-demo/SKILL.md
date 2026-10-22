---
name: hf-publish-demo
description: Skill 9 of 9 for TinyOps. Use to publish everything for the Lexsi hackathon submission — HF model and dataset repos with cards, a Gradio HF Space with live and replay modes, the HF Article, the GitHub README, demo video script, Lexsi feedback issues, and the final submission checklist before Nov 14 EOD.
---

# Skill 9 — Publish, demo & submit

## Goal
A complete, public, reproducible submission that meets every hackathon requirement and reads well to Lexsi's judges.

## Inputs
- Everything in `results/final/`, adapters on the Hub, `docs/` (METRICS, BASELINE_FINDINGS, FAILURE_ANALYSIS, DECISIONS).

## Files
```
space/app.py  space/requirements.txt  space/README.md  space/replays/*.jsonl
hub/model_card_final.md  hub/model_card_sft.md  hub/dataset_card.md
src/tinyops/report/publish.py          # pushes cards/files to the Hub
docs/ARTICLE.md
docs/DEMO_SCRIPT.md
docs/LEXSI_FEEDBACK.md
README.md
```

## 9.1 Hugging Face repos
| repo | contents |
|---|---|
| `<user>/tinyops-tiny-aya` (model) | final adapter, `lexsi_provenance.json`, model card |
| `<user>/tinyops-tiny-aya-sft` (model) | SFT adapter + card |
| `<user>/tinyops-data` (dataset) | `scenarios/{train,val,test,test_injection,test_ood}.jsonl`, `TEST_HASHES.txt`, gold trajectories, `baselines/`, `final/episodes`, `final/metrics`, `training_logs/` (trajectories.jsonl, reward CSVs), dataset card |
| `<user>/tinyops-demo` (Space) | Gradio app |
`report/publish.py` uses `huggingface_hub.HfApi` (`create_repo`, `upload_folder`, `upload_file`) and is idempotent.

**Model card (final)** sections: summary; base model + licence (CC-BY-NC, Cohere AUP); training (data counts, SFT + Stage B method, hyperparameters, GPU hours); evaluation table copied from `results/final/tables.md` by script (not by hand); safety results on `test_injection`; limitations (simulated cluster, keyword-based matcher with measured agreement, English only, one model size); how to load:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "<user>/tinyops-tiny-aya")
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
```
Credit AgentTune (LSAL v1.1), AuditKIT (LSAL v1.2), Tiny Aya.

**Dataset card:** what KubeSim is, the 9 failure types, splits and sizes, injection families, generation seed and code version, test-lock policy, licence of the generated data (your choice, e.g. CC-BY-4.0, noting it's synthetic), limitations.

## 9.2 Space — `space/app.py` (Gradio Blocks)
UI:
- Left: scenario picker (dropdown of ~20 curated test_injection/test examples + "random val scenario"), toggle "inject attack", model radio (Base Tiny Aya / TinyOps), mode radio (Live / Replay).
- Center: streaming transcript — each step shows the tool call (code-formatted), a ✅/⚠️/❌ badge (valid / schema error / unsafe), and the truncated tool result.
- Right: final diagnosis, evidence, verdict (correct / wrong), safety badge (🟢 no unsafe action / 🔴 unsafe attempt blocked), steps used.
- Footer: headline numbers table loaded from `results/final/report.json` (copied into the Space at build).
Modes:
- **Replay** (default, always works): plays pre-recorded episodes from `space/replays/` (base vs final on the same scenarios).
- **Live:** runs KubeSim + the model. Use ZeroGPU (`@spaces.GPU`) if available on the account, else a paid small GPU from credits; if neither, hide Live.
Requirements pinned in `space/requirements.txt`. Space README front-matter: `title`, `sdk: gradio`, `app_file: app.py`, licence, links to model/dataset/article.

## 9.3 HF Article — `docs/ARTICLE.md` → publish on HF (blog/community article)
Structure (target 1,500–2,500 words, numbers only from `results/final/`):
1. **TL;DR** — 3 bullets with the headline before→after numbers and the safety result.
2. **Why small models fail at tool use** — 2–3 real baseline transcripts (from BASELINE_FINDINGS).
3. **KubeSim** — environment, 9 failure types, difficulty, distractors, injection suite, locked/OOD splits (diagram).
4. **Metrics and why** — the TinyScore table (condensed from METRICS.md), why grounding and unsafe rate matter, matcher-agreement number.
5. **Approach** — why SFT first (format), why GRPO/DPO second (efficiency, grounding, safety), reward design and the anti-hacking checks; what AgentTune and AuditKIT did in the pipeline.
6. **Results** — main table with CIs and p-values; plots; per-type heatmap; OOD gap.
7. **Injection robustness** — unsafe rate by family, base vs final; over-refusal control.
8. **What still breaks** — from FAILURE_ANALYSIS.md.
9. **Cost (ROI)** — table + break-even, with price sources and dates.
10. **Limitations & next steps** — simulated env, matcher, single model/language, ideas (real cluster replay, Aya Expanse 8B student).
11. **Reproduce** — exact commands, repo/model/dataset/Space links.
12. **Acknowledgements** — Lexsi Labs (AgentTune, AuditKIT), Cohere Labs (Tiny Aya), Hugging Face.

## 9.4 GitHub README
Sections: one-paragraph pitch; results table (generated from report.json by `make_tables.py --readme`); architecture diagram (mermaid); quickstart (CPU: generate data, run oracle, score); reproduce training (notebooks in order); repo map; links (Space, model, dataset, article); licence notes (code licence of your choice, e.g. MIT, plus third-party licences); citation of AgentTune/AuditKIT/Tiny Aya.

## 9.5 Demo video — `docs/DEMO_SCRIPT.md` (2–3 min)
0:00 problem (small models + tools + safety) → 0:25 KubeSim in 1 slide → 0:45 replay: base model fumbles an incident (malformed calls/loop) → 1:15 same incident with TinyOps → 1:40 injection scenario: base obeys/attempts, TinyOps ignores and still diagnoses → 2:05 results table + cost → 2:30 limitations + links. Record with OBS; upload unlisted to YouTube; link in Space and article.

## 9.6 Feedback to Lexsi — `docs/LEXSI_FEEDBACK.md`
List every AgentTune/AuditKIT issue hit during the build (from DECISIONS.md): minimal repro, expected vs actual, version/commit, suggested fix. File each as a GitHub issue (security-sensitive ones via SECURITY.md instead). Link the issues in the article's acknowledgements. Keep the tone constructive.

## 9.7 Submission checklist (submit by Nov 14, aim for Nov 13 night)
- [ ] Repo public, CI green, README complete.
- [ ] Model(s), dataset (with logs + provenance) and Space public on HF; Space loads in replay mode.
- [ ] HF Article published; ≥ 2 before/after metrics with CIs (we have many) and the write-up on metric/data/model choices.
- [ ] ROI section included (optional but preferred).
- [ ] All links cross-checked (article ↔ Space ↔ model ↔ dataset ↔ GitHub).
- [ ] HF usernames in the submission match the registration form (credits).
- [ ] Submission form filled; confirmation saved (screenshot) in `docs/`.
- [ ] Post on LinkedIn/X tagging Lexsi Labs, Cohere Labs, Hugging Face; share in Discord.

## If shortlisted (Nov 16–21, $200 credits)
Prioritize, in order: (1) larger test set for tighter CIs, (2) Aya Expanse 8B as student or teacher, (3) more failure types (network policy, PVC pending, HPA thrash), (4) live mode on a GPU for Demo Day. Re-run skill 8 for any new model; keep the old numbers too.

## Exit gate
- [ ] Every checklist item ticked; submission confirmation saved.
- [ ] Commit `skill-9: publish, demo, article, submission`.
