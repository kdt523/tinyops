---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: peft
datasets: [<user>/tinyops-data]
tags: [lora, tool-use, kubernetes, agent, safety, prompt-injection, aya]
---
# TinyOps: Tiny Aya for safe Kubernetes incident response

LoRA adapter(s) that teach **CohereLabs/tiny-aya-global** (3.35B) to investigate a Kubernetes incident with
tool calls, submit a grounded diagnosis, and **ignore instructions hidden in logs, events and configs**.
Everything is trained and evaluated in a **simulated** cluster (KubeSim); nothing here has been run
against a real cluster.

## Model
- Base: Tiny Aya Global, licence CC-BY-NC (gated) and Cohere's acceptable-use policy.
- Method: _final model not chosen yet_.
- Training data: 1215 generated incidents (9 failure types x 3 difficulties, 15-20% with an injected
  instruction), gold trajectories from a scripted oracle. Hyper-parameters: `configs/sft.yaml`,
  `configs/grpo.yaml`, `configs/dpo.yaml` in the code repo. Compute: _not recorded yet_.

## Evaluation (locked test splits, greedy decoding, KubeSim, 8 steps max)
Tables are copied by `python -m tinyops.report.cards` from `results/final/tables.md`.

### test (270 incidents, no injections)
_Pending: the locked final evaluation (skill 8) has not been run yet._

### test_injection (162 incidents, every attack family)
_Pending: the locked final evaluation (skill 8) has not been run yet._

### Out of distribution (test_ood: new names, registry and alert wording)
_Pending: the locked final evaluation (skill 8) has not been run yet._

### Paired significance tests
_Pending: the locked final evaluation (skill 8) has not been run yet._

## Limitations
Simulated environment only; the diagnosis matcher is keyword-based (its agreement with an LLM judge is reported
in the repo's `docs/METRICS.md` once measured); English only; a single model size; injection families are the
six listed in the dataset card, so robustness to other attacks is untested.

## How to load
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "<user>/tinyops-tiny-aya")
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
```
Use the system prompt and tool-call format from `tinyops.policies.prompt` (prompt version `v1`).

## Licence and credits
Research and hackathon use only. Base model: Tiny Aya (CC-BY-NC, Cohere Labs). Training: AgentTune (LSAL v1.1).
Evaluation: AuditKIT (LSAL v1.2) by Lexsi Labs. Hosting: Hugging Face.
