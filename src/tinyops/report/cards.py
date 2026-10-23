"""Hub cards generated from files: final model card and dataset card.

    python -m tinyops.report.cards            # writes hub/model_card_final.md and hub/dataset_card.md

Result tables are copied from results/final/tables.md by this script, never typed. Missing results
produce an explicit "pending" notice, not a number.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Annotated

import typer

from tinyops.scenarios.failure_types import FAILURE_TYPE_NAMES
from tinyops.scenarios.injection import FAMILIES
from tinyops.utils.io import read_json
from tinyops.utils.paths import DATA, RESULTS, ROOT

PENDING = "_Pending: the locked final evaluation (skill 8) has not been run yet._"
SPLIT_ORDER = ["train", "val", "test", "test_injection", "test_ood"]


def tables_section(tables_md: str | None, heading: str) -> str:
    """The `## heading` block of results/final/tables.md (up to the next `## `)."""
    if not tables_md:
        return PENDING
    m = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", tables_md, re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else PENDING


def gpu_hours(compute_log: Path) -> str:
    if not compute_log.exists():
        return "_not recorded yet_"
    rows = list(csv.DictReader(compute_log.open(encoding="utf-8")))
    train = [float(r["hours"]) for r in rows if r["run"].endswith("_train")]
    return f"{sum(train):.2f} GPU-hours (training runs in `results/compute_log.csv`)" if train else "_not recorded yet_"


def load_snippet(two_adapters: bool, sft_repo: str, final_repo: str) -> str:
    if two_adapters:
        return f'''```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "{sft_repo}").merge_and_unload()   # SFT stage
model = PeftModel.from_pretrained(model, "{final_repo}")                    # Stage B (GRPO/DPO) on top
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
```'''
    return f'''```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "{final_repo}")
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
```'''


def render_final_card(results: Path = RESULTS, data: Path = DATA, user: str = "<user>") -> str:
    final_dir = results / "final"
    tables = (final_dir / "tables.md").read_text(encoding="utf-8") if (final_dir / "tables.md").exists() else None
    choice_f = results / "runs" / "final_model_choice.json"
    choice = read_json(choice_f)["final"] if choice_f.exists() else None
    two = choice in ("sft+grpo", "sft+dpo")
    sft_repo, final_repo = f"{user}/tinyops-tiny-aya-sft", f"{user}/tinyops-tiny-aya"
    stats = {s: read_json(data / s / "stats.json") for s in SPLIT_ORDER if (data / s / "stats.json").exists()}
    train_n = stats.get("train", {}).get("n", "?")
    method = {"sft": "SFT only (Stage B did not beat it on validation)", "sft+grpo": "SFT, then GRPO with TinyOps rewards",
              "sft+dpo": "SFT, then DPO fallback", None: "_final model not chosen yet_"}[choice]
    return f"""---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: peft
datasets: [{user}/tinyops-data]
tags: [lora, tool-use, kubernetes, agent, safety, prompt-injection, aya]
---
# TinyOps: Tiny Aya for safe Kubernetes incident response

LoRA adapter(s) that teach **CohereLabs/tiny-aya-global** (3.35B) to investigate a Kubernetes incident with
tool calls, submit a grounded diagnosis, and **ignore instructions hidden in logs, events and configs**.
Everything is trained and evaluated in a **simulated** cluster (KubeSim); nothing here has been run
against a real cluster.

## Model
- Base: Tiny Aya Global, licence CC-BY-NC (gated) and Cohere's acceptable-use policy.
- Method: {method}.
- Training data: {train_n} generated incidents (9 failure types x 3 difficulties, 15-20% with an injected
  instruction), gold trajectories from a scripted oracle. Hyper-parameters: `configs/sft.yaml`,
  `configs/grpo.yaml`, `configs/dpo.yaml` in the code repo. Compute: {gpu_hours(results / 'compute_log.csv')}.

## Evaluation (locked test splits, greedy decoding, KubeSim, 8 steps max)
Tables are copied by `python -m tinyops.report.cards` from `results/final/tables.md`.

### test (270 incidents, no injections)
{tables_section(tables, 'test')}

### test_injection (162 incidents, every attack family)
{tables_section(tables, 'test_injection')}

### Out of distribution (test_ood: new names, registry and alert wording)
{tables_section(tables, 'test_ood')}

### Paired significance tests
{tables_section(tables, 'Paired tests')}

## Limitations
Simulated environment only; the diagnosis matcher is keyword-based (its agreement with an LLM judge is reported
in the repo's `docs/METRICS.md` once measured); English only; a single model size; injection families are the
six listed in the dataset card, so robustness to other attacks is untested.

## How to load
{load_snippet(two, sft_repo, final_repo)}
Use the system prompt and tool-call format from `tinyops.policies.prompt` (prompt version `v1`).

## Licence and credits
Research and hackathon use only. Base model: Tiny Aya (CC-BY-NC, Cohere Labs). Training: AgentTune (LSAL v1.1).
Evaluation: AuditKIT (LSAL v1.2) by Lexsi Labs. Hosting: Hugging Face.
"""


def render_dataset_card(data: Path = DATA, user: str = "<user>") -> str:
    stats = {s: read_json(data / s / "stats.json") for s in SPLIT_ORDER if (data / s / "stats.json").exists()}
    rows = ["| Split | Incidents | Injected | Purpose |", "|---|---|---|---|"]
    purpose = {"train": "SFT / RL training", "val": "all model and checkpoint selection", "test": "final eval, no injections",
               "test_injection": "final eval, every injection family", "test_ood": "final eval, unseen names/alerts"}
    for s, st in stats.items():
        rows.append(f"| `{s}` | {st['n']} | {st['n_injected']} | {purpose[s]} |")
    hashes = (data / "TEST_HASHES.txt").read_text(encoding="utf-8").strip() if (data / "TEST_HASHES.txt").exists() else "_not generated_"
    return f"""---
license: cc-by-4.0
task_categories: [text-generation]
language: [en]
tags: [kubernetes, agents, tool-use, synthetic, safety, prompt-injection]
---
# {user}/tinyops-data

Synthetic incident-response data for TinyOps: scenarios from **KubeSim**, a deterministic fake Kubernetes cluster,
with hidden ground truth, gold (oracle) trajectories, baseline and final evaluation episodes, and training logs.
All data is machine-generated; there is no real cluster data, customer data or personal data.

## KubeSim
Ten tools (`list_pods`, `describe_pod`, `get_events`, `get_logs`, `get_metrics`, `check_service`, `get_configmap`,
`restart_deployment`, `delete_namespace`, `submit_diagnosis`); `delete_namespace` is always denied and counts as an
unsafe action. Tool results are kubectl-like text, at most 1,500 characters.

## Failure types ({len(FAILURE_TYPE_NAMES)})
{', '.join(f'`{t}`' for t in FAILURE_TYPE_NAMES)}. Difficulties: easy (one namespace, no distractors), medium (healthy
neighbours and noisy logs), hard (second namespace, buried key line, misleading warning, unrelated events).

## Injection suite
Families: {', '.join(f'`{f}`' for f in FAMILIES)} (`benign_lookalike` is a harmless control that measures
over-refusal). Payloads sit in a log line, an event message or a ConfigMap value; the real fault stays intact.

## Splits (generated with seed 1234, generator version v1)
{chr(10).join(rows)}

`test*` splits are **locked**: generated once, SHA-256 hashed, never used for training, prompt tuning,
checkpoint selection or hyper-parameters. Hashes:
```
{hashes}
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
"""


app = typer.Typer(add_completion=False)


@app.command()
def main(
    results: Annotated[Path, typer.Option()] = RESULTS,
    data: Annotated[Path, typer.Option()] = DATA,
    user: Annotated[str, typer.Option(help="HF username")] = "<user>",
    out: Annotated[Path, typer.Option()] = ROOT / "hub",
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_card_final.md").write_text(render_final_card(results, data, user), encoding="utf-8")
    (out / "dataset_card.md").write_text(render_dataset_card(data, user), encoding="utf-8")
    typer.echo(f"wrote {out / 'model_card_final.md'} and {out / 'dataset_card.md'}")


if __name__ == "__main__":
    app()
