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
