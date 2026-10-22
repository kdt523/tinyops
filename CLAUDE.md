# TinyOps — Project Brain (CLAUDE.md)

You are building **TinyOps**: post-training Cohere **Tiny Aya Global** (3.35B) to call Kubernetes incident-response tools **correctly, efficiently and safely**, with Lexsi Labs' **AgentTune** (training) and **AuditKIT** (evaluation). Target: **Track 3 — Agentic Fine-tuning**, Lexsi AI Safety & Alignment Hackathon (build Nov 1–14 2026, submit by EOD Nov 14).

The work is split into **9 skills** in `.claude/skills/`. Run them **in order**. Each skill has a goal, inputs, exact tasks, file contracts, tests and an **exit gate**. Never start a skill until the previous skill's exit gate passes.

| # | Skill | What it produces | When |
|---|---|---|---|
| 1 | `tinyops-setup` | env check, repo scaffold, CI, API notes | Oct 22–Nov 1 |
| 2 | `kubesim-environment` | fake K8s cluster + 10 tools + safety gate + episode runner | Nov 1–2 |
| 3 | `incident-scenarios` | 9 failure generators, injection suite, locked splits, gold trajectories | Nov 2–3 |
| 4 | `tinyscore-metrics` | deterministic scorer, AuditKIT bridge, stats | Nov 3–4 |
| 5 | `baseline-eval` | model policies, parser, Colab notebook, "before" numbers | Nov 4–5 |
| 6 | `sft-training` | SFT LoRA adapter on gold trajectories | Nov 5–7 |
| 7 | `grpo-training` | reward functions + GRPO (or DPO fallback) adapter | Nov 7–10 |
| 8 | `final-eval-report` | locked-test results, significance, plots, failure analysis, ROI | Nov 10–11 |
| 9 | `hf-publish-demo` | HF model/dataset/Space, article, README, submission | Nov 12–14 |

---

## Non-negotiable rules (apply in every skill)

1. **Verify before you call.** AgentTune and AuditKIT are new and change weekly. Before using any function/class, open the installed source and confirm its name, signature, arguments and return type:
   ```bash
   python -c "import agenttune, os; print(os.path.dirname(agenttune.__file__))"
   python -c "import auditkit, os; print(os.path.dirname(auditkit.__file__))"
   grep -rn "def create_agentic_trainer" $(python -c "import agenttune,os;print(os.path.dirname(agenttune.__file__))")
   ```
   If reality differs from a skill, **follow the source**, adapt, and log it in `docs/DECISIONS.md` (date, what the skill said, what the source does, what you did).
2. **No fabricated numbers.** Every metric in README/article/model card must be read from a file in `results/` produced by a script in this repo. If a run fails, say so. Never "estimate" a result.
3. **CPU path always works.** Simulator, scenarios, scoring, rewards and all tests run on CPU with no model download. GPU is only for Tiny Aya inference/training (Colab/Kaggle/HF).
4. **Locked test set.** `data/test*/` is generated once (skill 3), SHA-256 hashed, and never used for training, prompt tuning, checkpoint selection or hyperparameters. Use `data/val/` for all of that. Skill 8 is the only place test splits are evaluated with a trained model.
5. **Determinism.** Every generator/evaluation takes `--seed`; default seed `1234`. Greedy decoding (temperature 0) for evaluation.
6. **Tests first for logic.** Every module under `src/tinyops/` (except notebooks) gets pytest tests. `pytest -q` must be green before a skill's exit gate.
7. **Small commits.** Commit at the end of each skill: `skill-N: <summary>`. Never commit secrets (`.env`, HF tokens, API keys).
8. **Record environment.** `python -m tinyops.utils.envinfo` writes `results/env.json` (python, package versions, git commit, GPU). Run it at the start of every GPU session.
9. **Known AgentTune pitfalls — avoid:** don't use `ToolRegistry`/`auto_register_builtins()` (imports langchain_community tools); don't rely on the CLI for tool training (no `--tools` flag); don't use `RougeMetric`, `BleuMetric` or `DistilledJudge` (fail silently).
10. **Hackathon rules:** code written Nov 1–14; use a Cohere model; host models/adapters/logs/datasets on Hugging Face; show >1 before/after metric; publish an HF Article; ROI comparison is "optional but preferred".
11. **Licenses:** Tiny Aya = CC-BY-NC (gated). AgentTune = LSAL v1.1, AuditKIT = LSAL v1.2 (source-available, non-commercial). Research/hackathon use only — state this in README and model cards.

## Coding conventions
- Python ≥ 3.12, package `tinyops` under `src/`. Pydantic v2 for data models. Type hints everywhere. `ruff` for lint/format (line length 100).
- CLIs with `typer`: `python -m tinyops.<module> --help` must work.
- Logging with `rich`/`logging`, never bare `print` in library code.
- All file I/O through `tinyops.utils.io` (`read_jsonl`, `write_jsonl`, `write_json`, `sha256_file`).
- Paths relative to repo root via `tinyops.utils.paths.ROOT`.

## Glossary
- **Scenario / incident**: one generated cluster state + alert + hidden ground truth.
- **Episode**: one run of a policy on a scenario (list of steps until `submit_diagnosis`, max steps, or parse-failure stop).
- **Policy**: anything that maps conversation → next model text (scripted oracle, Tiny Aya, Tiny Aya+LoRA, teacher API model).
- **Gold trajectory**: an episode from the scripted oracle (or a filtered teacher episode) used for SFT.
- **TinyScore**: our deterministic per-episode metrics (skill 4).

## How to drive this with Claude Code
1. Copy `CLAUDE.md` and the `.claude/` folder into the root of your `tinyops` repo.
2. Start Claude Code in the repo and say: **"Read CLAUDE.md, then run the `tinyops-setup` skill."**
3. After each skill's exit gate passes, say: **"Run the `<next-skill>` skill."** (order: tinyops-setup → kubesim-environment → incident-scenarios → tinyscore-metrics → baseline-eval → sft-training → grpo-training → final-eval-report → hf-publish-demo).
4. GPU skills (5–8) produce notebooks; run them on Colab/Kaggle/HF, then upload the `results/` files back and ask Claude to continue from them.
