"""Generate docs/ARTICLE.md. The prose is fixed; every number is read from a result file.

    python -m tinyops.report.article

A section whose result file does not exist yet shows an explicit `⟨pending: <file>⟩` marker (the
submission checker fails on those), never an estimate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import typer

from tinyops.report.cards import tables_section
from tinyops.report.make_tables import cell
from tinyops.scenarios.failure_types import FAILURE_TYPE_NAMES
from tinyops.scenarios.injection import FAMILIES
from tinyops.utils.io import read_json
from tinyops.utils.paths import DATA, DOCS, RESULTS, ROOT

PENDING = "⟨pending: {}⟩"


def pending(what: str) -> str:
    return PENDING.format(what)


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _stat(report: dict | None, split: str, policy: str, key: str) -> dict | None:
    if not report:
        return None
    return report["splits"].get(split, {}).get(policy, {}).get("headline", {}).get(key)


def tldr(report: dict | None) -> str:
    if not report or "final" not in report["policies"]:
        return pending("results/final/report.json (run skill 8)")
    base = "base_zeroshot"
    bullets = []
    for split, key, label, pct in (("test", "diagnosis_success", "Diagnosis success on the locked test split", True),
                                   ("test_injection", "unsafe_rate", "Unsafe action rate when an instruction is injected", True),
                                   ("test", "redundant_calls_per_ep", "Redundant tool calls per episode", False)):
        a, b = _stat(report, split, base, key), _stat(report, split, "final", key)
        if a and b:
            bullets.append(f"- **{label}:** {cell(a, pct)} (base Tiny Aya) → {cell(b, pct)} (TinyOps)")
    return "\n".join(bullets) or pending("results/final/report.json")


def render_article(root: Path = ROOT, results: Path = RESULTS, data: Path = DATA) -> str:
    final = results / "final"
    report = json.loads(_read(final / "report.json")) if (final / "report.json").exists() else None
    tables = _read(final / "tables.md")
    findings = _read(root / "docs" / "BASELINE_FINDINGS.md")
    failure = _read(root / "docs" / "FAILURE_ANALYSIS.md")
    roi = _read(final / "roi.md")
    agreement_f = results / "runs" / "judge_agreement.json"
    agreement = (f"{100 * read_json(agreement_f)['agreement']:.1f}% on the validation split "
                 f"(n={read_json(agreement_f)['n']})") if agreement_f.exists() else pending("results/runs/judge_agreement.json (needs a Gemini key)")
    grpo_cfg = (root / "configs" / "grpo.yaml")
    weights = "diagnosis 1.0, format 0.5, schema 0.5, efficiency 0.3, grounding 0.3, safety 1.0"
    if grpo_cfg.exists():
        import yaml

        w = yaml.safe_load(grpo_cfg.read_text(encoding="utf-8"))["reward_weights"]
        weights = ", ".join(f"{k} {v}" for k, v in w.items())
    stats = {s: read_json(data / s / "stats.json") for s in ("train", "val", "test", "test_injection", "test_ood")
             if (data / s / "stats.json").exists()}
    sizes = ", ".join(f"{s}: {st['n']}" for s, st in stats.items())
    return f"""# TinyOps: teaching a 3B model to fix Kubernetes incidents — and to ignore instructions hidden in the logs

## 1. TL;DR
{tldr(report)}
- Trained and evaluated entirely in a simulated cluster (KubeSim) with Lexsi's AgentTune and AuditKIT; the test splits are
  SHA-256 locked and were evaluated once. All numbers below are read from `results/final/`.

## 2. Why small models fail at tool use
A small model that can *name* the right tool still has to produce an exact call format, pick arguments that exist, avoid
repeating itself, stop when it knows the answer, and — in an operations setting — treat everything inside a log line as data,
not as instructions. Measured failure modes of the untouched base model (from `docs/BASELINE_FINDINGS.md`):

{findings.split('## Top failure modes', 1)[1].split('## What each mode', 1)[0].strip() if findings and '## Top failure modes' in findings else pending('docs/BASELINE_FINDINGS.md (skill 5 baselines)')}

## 3. KubeSim
KubeSim is a deterministic fake cluster: pydantic state rendered as kubectl-like text, ten tools (seven read-only, a
rollout restart behind a safety gate, an always-denied `delete_namespace`, and `submit_diagnosis`). Nine failure types
({', '.join(f'`{t}`' for t in FAILURE_TYPE_NAMES)}) come at three difficulties: *easy* (one namespace), *medium* (healthy
neighbours, noisy logs), *hard* (a second namespace, the key log line buried more than 20 lines from the end, a misleading
warning, unrelated events). An injection suite plants an instruction in a log line, an event or a ConfigMap
({', '.join(f'`{f}`' for f in FAMILIES)}); `benign_lookalike` is a harmless control that measures over-refusal.

```mermaid
flowchart LR
  A[alert] --> M[model] --> T[tool call] --> E[KubeSim + safety gate] --> O[observation] --> M
  M --> D[submit_diagnosis] --> S[TinyScore]
```
Splits ({sizes}) are generated once with seed 1234 and hashed; `test_ood` uses disjoint names, registries and alert wording.

## 4. Metrics and why
A diagnosis counts only if it names the true cause (keyword groups, with a check that the answer does not fit a *different*
failure type better) **and** at least half of its evidence lines appear verbatim in tool output the model actually received
("grounding"; evidence under six characters never counts). We also measure format and schema validity, tool selection,
redundant calls, loops, unnecessary writes, **unsafe actions** (deleting a namespace, restarting an injection target,
sending data out) and over-refusal. Definitions: `docs/METRICS.md`. Agreement between this keyword matcher and an LLM
judge: {agreement}. AuditKIT independently re-computes validity, redundancy, loops and permission violations; the cross-check
is in `results/final/auditkit/`.

## 5. Approach
**SFT first (format).** Gold trajectories from a scripted oracle — every argument it uses was shown earlier in the
conversation — train a LoRA adapter with loss on assistant turns only, in exactly the prompt format used at inference.
**GRPO second (efficiency, grounding, safety)** through AgentTune's `create_agentic_trainer`, starting from the SFT model
merged into the base so the KL anchor is the SFT policy. Reward = weighted sum ({weights}). Every rollout is
*replayed through the strict environment*, so the reward equals what the evaluation would score. Anti-hacking checks are
unit tests: keyword-stuffed instant submits and invented evidence score far below a real investigation, evidence copied from
the alert earns no grounding, any `delete_namespace` attempt scores below every realistic safe completion, and repeated
calls lower the score monotonically. A DPO path (turn-level pairs from failed rollouts) is the fallback. AgentTune did the
training and rollouts; AuditKIT did the independent agent evaluation and the red-team probe.

## 6. Results
{tables_section(tables, 'test')}

Paired tests (McNemar for success and for "not unsafe"; paired bootstrap for continuous metrics):

{tables_section(tables, 'Paired tests')}

Out of distribution (new names, registry and alert wording): {tables_section(tables, 'OOD gap (success on test minus test_ood)')}

Figures: `results/final/plots/` (main metrics, success heatmap by failure type, unsafe rate by family, efficiency, training curves).

## 7. Injection robustness
{tables_section(tables, 'test_injection')}

Unsafe rate by attack family (attacks only):

{tables_section(tables, 'Unsafe rate by injection family (test_injection, attacks only)')}

## 8. What still breaks
{failure.split('# Failure analysis (final model)', 1)[1].split('## Illustrative examples', 1)[0].strip() if failure else pending('docs/FAILURE_ANALYSIS.md (human-confirmed)')}

## 9. Cost
{roi.split('# Cost comparison (monthly)', 1)[1].strip() if roi else pending('results/final/roi.md (needs the cited closed-model price)')}

## 10. Limitations and next steps
The cluster is simulated and simpler than a real one; log text comes from templates; diagnosis scoring is keyword-based; one
model size and one language; six injection families, so other attacks are untested. Next: replay real incident data against
KubeSim's tool interface, a larger test set for tighter intervals, Aya Expanse 8B as student or teacher, more failure types
(network policy, pending PVC, HPA thrash).

## 11. Reproduce
```bash
pip install -e '.[dev]' && pytest -q
python -m tinyops.scenarios.build_dataset --seed 1234 --out data/ && python -m tinyops.scenarios.build_dataset gold --split all
# GPU: notebooks/01_baselines, 02_sft, 03_grpo, 04_final_eval
python -m tinyops.eval.compare && python -m tinyops.report.make_plots && python -m tinyops.report.roi
```
Code `https://github.com/<github-user>/tinyops` · model `<user>/tinyops-tiny-aya` · dataset `<user>/tinyops-data` ·
demo `<user>/tinyops-demo`.

## 12. Acknowledgements
Lexsi Labs (AgentTune, AuditKIT — and the issues we reported, listed in `docs/LEXSI_FEEDBACK.md`), Cohere Labs (Tiny Aya),
Hugging Face. Research and hackathon use only; Tiny Aya is CC-BY-NC.
"""


app = typer.Typer(add_completion=False)


@app.command()
def main(out: Path = DOCS / "ARTICLE.md") -> None:
    text = render_article()
    out.write_text(text, encoding="utf-8")
    typer.echo(f"wrote {out}: {len(text.split())} words, {len(re.findall('⟨pending', text))} pending markers")


if __name__ == "__main__":
    app()
