---
name: tinyops-setup
description: Skill 1 of 9 for TinyOps. Use to set up accounts, the dev environment (WSL2 + Python 3.12), verify AgentTune/AuditKIT/Tiny Aya APIs, and scaffold the tinyops repo with CI. Run before any other TinyOps skill.
---

# Skill 1 — Setup, API verification & scaffold

## Goal
A working dev environment, confirmed knowledge of the real AgentTune/AuditKIT APIs, and an installable, tested, CI-checked empty `tinyops` package.

## Inputs
- Nothing from earlier skills.
- The human's accounts (HF, Gemini, Kaggle) and a Windows 11 laptop (12 GB RAM, 4 GB GPU).

## Part A — Before Nov 1 (no project code committed)

### A1. Accounts checklist (the human does these; Claude prints the checklist and waits for confirmation)
- [ ] Register at https://hackathon.lexsi.ai/signup/ — Track C "Agentic Fine-tuning". Use the **same HF username** you'll publish with (credits are matched on it).
- [ ] Join Discord (https://discord.gg/dtEDQ2Z3eg), get the hackathon role, introduce yourself in `#hackathon-lexsi`.
- [ ] RSVP the Oct 31 kickoff (Luma link on the site); attend online; write down any Track 3 sample problem statements.
- [ ] Hugging Face: accept licences for `CohereLabs/tiny-aya-global` (required) and `CohereLabs/aya-expanse-8b` (optional baseline). Create a **write** token.
- [ ] Redeem the $20 HF credits when the email arrives (they expire 30 days after redeeming).
- [ ] Gemini API key (closed-model reference + optional teacher).
- [ ] Kaggle account with phone verification (GPU quota).

### A2. Local environment (Windows 11 → WSL2)
AgentTune needs Python 3.12+ and Linux-friendly deps, so use WSL2.
```bash
# PowerShell (admin):  wsl --install -d Ubuntu-24.04   then reboot
# inside Ubuntu:
sudo apt update && sudo apt install -y git build-essential python3.12 python3.12-venv
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
mkdir -p ~/work && cd ~/work
git clone https://github.com/Lexsi-Labs/AgentTune.git
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch only on the laptop
uv pip install -e "./AgentTune[dev]"
uv pip install "auditkit[transformers]"
hf auth login   # or: huggingface-cli login
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
- If AgentTune's pinned torch conflicts with the CPU wheel, install AgentTune first and let it resolve torch, then confirm CUDA is not required. Record what happened.
- WSL2 memory: create `C:\Users\<you>\.wslconfig` with `[wsl2]\nmemory=8GB\nswap=8GB` so installs don't get killed.

### A3. Verify AgentTune on CPU — write findings to `notes/api_notes.md`
1. Run the README's GPU-free spine quickstart (`Project`, `DictToolHarness`, `ReActStrategy`) and confirm `proj.infer`, `proj.evaluate`, `proj.evaluate_agentic` run.
2. `cd AgentTune && pytest -q` → note failures (don't fix them now; they're possible contributions).
3. Read the source and answer each question below in the notes, with file path + line numbers:
   - **Trainer factory** (`agenttune/core/backend_factory.py`): full signature of `create_agentic_trainer`; which kwargs go to the TRL config vs the trainer; how `tools` (python callables) are turned into schemas; meaning of `tools_fallback_prompt`, `tool_result_format`, `max_steps_per_turn`, `trajectories_file`, `peft_config`.
   - **Rollout loop** (`agenttune/agentic/…`): how a tool call is parsed from model text (fallback list format, Cohere `<|START_ACTION|>`); what happens on a parse failure; how tool results are fed back; **how per-example dataset columns reach (a) tools and (b) reward functions**; whether reward functions can see the full multi-turn trajectory or only the final completion.
   - **Rewards** (`agenttune/agentic/rewards/builtin_rewards.py`): the names in `REWARD_REGISTRY`, the expected reward function signature, and how `combine_rewards` weights things.
   - **Distillation** (`create_distill_trainer`): accepted input shapes (`demonstrations`, `trajectories`, `teacher_engine`+`tasks`) and whether multi-turn tool trajectories are supported.
   - **EventLog**: constructor, light vs full tier fields, how to build one by hand.
   - **The exact Tiny Aya fallback prompt text and output format.** Copy it **verbatim** into the notes. Every later skill depends on it.
4. Read `docs/community/known-issues.md`, `CONTRIBUTING.md` and `CHANGELOG.md`.

### A4. Verify AuditKIT on CPU — add to `notes/api_notes.md`
1. Run the README quickstart (`ak.Sample`, `ak.evaluate` with a lambda model) and print `result.summary()`.
2. Read `auditkit/agent_eval/`: `episodes_from_agenttune`, `episode_from_eventlog`; how an outcome predicate/verifier is attached; which tool/trace metrics exist (names + exact definitions); the `auditkit agent` CLI flags.
3. Read `auditkit/redteam/`: `RedTeamRunner` constructor; probe families; detectors; how to pass a custom model callable (`list[str] -> list[str]`).
4. Read `ak.compare_models` and the significance test it uses.

### A5. Verify Tiny Aya on a GPU (Colab T4 or Kaggle)
Scratch notebook (not committed):
```python
!pip -q install -U transformers accelerate
from huggingface_hub import login; login()
import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
mid = "CohereLabs/tiny-aya-global"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float16, device_map="auto")
msgs = [{"role": "user", "content": "List 3 reasons a Kubernetes pod can be in CrashLoopBackOff."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
t = time.time(); out = model.generate(ids, max_new_tokens=200, do_sample=False)
gen = out[0][ids.shape[-1]:]
print(tok.decode(gen, skip_special_tokens=True))
print(f"{len(gen)/(time.time()-t):.1f} tok/s", torch.cuda.max_memory_allocated()/1e9, "GB")
```
Then install AgentTune in the GPU environment (needs Python 3.12; if Colab is older, use Kaggle or `uv venv --python 3.12`) and run the README's "Tool calling with Cohere models (Tiny Aya)" GRPO example with `max_steps=2`. Confirm `trajectories.jsonl` exists and loads with `auditkit.agent_eval.episodes_from_agenttune`. Record VRAM, seconds/step, Python version and tokens/sec.

### Part A exit gate
- [ ] Account checklist done.
- [ ] `notes/api_notes.md` answers every question in A3–A4 with file paths.
- [ ] Tiny Aya generates on GPU; the 2-step GRPO smoke run works and its output loads in AuditKIT.

## Part B — Nov 1: scaffold the repo

### B1. Layout
```
tinyops/
├── CLAUDE.md
├── .claude/skills/…              # these 9 skills
├── README.md                      # placeholder title + one line
├── pyproject.toml
├── .gitignore                     # .venv/ __pycache__/ runs/ *.safetensors .env wandb/ results/episodes/ data/**/*.parquet
├── .env.example                   # HF_TOKEN=  GEMINI_API_KEY=  HF_USER=
├── configs/                       # .gitkeep (filled in later skills)
├── data/                          # .gitkeep
├── docs/DECISIONS.md  docs/METRICS.md  docs/api_notes.md
├── notebooks/  results/  space/   # .gitkeep each
├── src/tinyops/
│   ├── __init__.py                # __version__ = "0.1.0"
│   ├── sim/ scenarios/ policies/ scoring/ train/ eval/ report/   # each with __init__.py
│   └── utils/{__init__,io,paths,envinfo,seed,log}.py
└── tests/test_smoke.py
```

### B2. `pyproject.toml`
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "tinyops"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "pydantic>=2.7", "jsonschema>=4.21", "typer>=0.12", "rich>=13", "pyyaml>=6",
  "pandas>=2.2", "numpy>=1.26", "scipy>=1.12", "matplotlib>=3.8", "python-dotenv>=1",
  "auditkit", "litellm",
]
[project.optional-dependencies]
train = ["agenttune @ git+https://github.com/Lexsi-Labs/AgentTune.git@<PINNED_COMMIT>",
         "transformers", "trl", "peft", "accelerate", "datasets", "bitsandbytes"]
space = ["gradio>=5"]
dev = ["pytest>=8", "pytest-cov", "ruff>=0.6"]

[tool.hatch.build.targets.wheel]
packages = ["src/tinyops"]
[tool.ruff]
line-length = 100
[tool.pytest.ini_options]
testpaths = ["tests"]
```
Replace `<PINNED_COMMIT>` with the AgentTune commit you verified in A3 (`git -C AgentTune rev-parse HEAD`).

### B3. Utilities (implement + test)
- `utils/io.py`: `read_jsonl(path) -> list[dict]`, `write_jsonl(path, rows: Iterable[dict])`, `append_jsonl(path, row)`, `read_json(path)`, `write_json(path, obj)` (indent 2, sorted keys, UTF-8), `sha256_file(path) -> str`. Parent directories are created automatically.
- `utils/paths.py`: `ROOT` = first parent folder containing `pyproject.toml`; `DATA`, `RESULTS`, `RUNS`, `CONFIGS`, `DOCS` as `Path`s.
- `utils/seed.py`: `set_seed(seed: int)` seeds `random`, `numpy` and `torch` (only if importable).
- `utils/envinfo.py`: typer app; `python -m tinyops.utils.envinfo` writes `results/env.json` = {python, platform, packages: versions of agenttune/auditkit/trl/transformers/torch/peft/accelerate/datasets (or null), git_commit, cuda_available, gpu_name, gpu_mem_gb, timestamp_utc}.
- `utils/log.py`: `get_logger(name)` using `rich.logging.RichHandler`, level from env `TINYOPS_LOG` (default INFO).

### B4. CI — `.github/workflows/ci.yml`
```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e '.[dev]'
      - run: ruff check .
      - run: pytest -q
```
CI never installs `[train]` (no torch/GPU in CI).

### B5. Docs
- Move `notes/api_notes.md` → `docs/api_notes.md`.
- `docs/DECISIONS.md` entry template:
  ```
  ## YYYY-MM-DD — <title>
  **Plan said:** …
  **Source/reality:** … (file:line)
  **Decision:** …
  ```
- `docs/METRICS.md`: stub heading (filled in skill 4).

## Tests (`tests/test_smoke.py`)
- Every subpackage imports.
- `write_jsonl` → `read_jsonl` round-trip preserves unicode and nested dicts.
- `sha256_file` is stable for the same content and changes when content changes.
- `ROOT / "pyproject.toml"` exists.
- `set_seed(1)` then `random.random()` gives the same value twice.
- `envinfo` writes valid JSON with the expected keys (run in a tmp dir).

## Exit gate
- [ ] `pip install -e '.[dev]'` works in WSL2; `pytest -q` green; `ruff check .` clean.
- [ ] CI green on GitHub (public repo, e.g. `github.com/kdt523/tinyops`).
- [ ] `docs/api_notes.md` committed with verified signatures and the verbatim Tiny Aya fallback format.
- [ ] Commit `skill-1: scaffold, utils, CI, API notes`.
