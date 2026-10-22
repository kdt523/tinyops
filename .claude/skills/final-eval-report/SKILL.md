---
name: final-eval-report
description: Skill 8 of 9 for TinyOps. Use once the final model is chosen, to run the single locked-test evaluation (test, test_injection, test_ood) for all policies, compute CIs and McNemar/paired-bootstrap tests, cross-check with AuditKIT and RedTeamRunner, make plots and tables, do the failure analysis, and compute the ROI comparison from measured numbers.
---

# Skill 8 — Final evaluation, failure analysis & ROI

## Goal
The official, reproducible before/after results on the locked test splits, plus an honest failure analysis and a cost comparison against a closed model.

## Inputs
- Final model choice (skill 7), SFT adapter, baseline results (skill 5), compute log.

## Files
```
src/tinyops/eval/final_eval.py
src/tinyops/eval/compare.py
src/tinyops/report/make_tables.py
src/tinyops/report/make_plots.py
src/tinyops/report/failure_analysis.py
src/tinyops/report/roi.py
configs/roi.yaml
notebooks/04_final_eval.ipynb
docs/FAILURE_ANALYSIS.md
tests/test_compare.py tests/test_roi.py
```

## 8.1 Run once on locked tests — `eval/final_eval.py`
1. Verify `TEST_HASHES.txt`; abort on mismatch.
2. Policies: `base_zeroshot`, `base_fewshot`, `sft`, `final` (if different from sft), `gemini` (from skill 5 results, test + test_injection only). Reuse skill 5 base-model test episodes if they were produced with the same prompt version and git-tracked code; otherwise re-run.
3. Splits: `test` (270), `test_injection` (162), `test_ood` (63). Greedy decoding, seed 1234, same `max_steps=8`.
4. Write `results/final/episodes/<policy>_<split>.jsonl` and `results/final/metrics/<policy>_<split>.json`.
5. Lock: after this run, record a `results/final/FINAL_RUN.json` (git commit, adapter revisions/hashes on the Hub, timestamp). Any re-run must be justified in DECISIONS.md (e.g. a bug fix) and re-run **all** policies.

## 8.2 Comparisons — `eval/compare.py`
- Main table (per split): every headline metric for each policy with 95% CI.
- Pairwise tests base_zeroshot → final and sft → final:
  - McNemar on `success` (test, test_ood) and on `unsafe` (test_injection).
  - Paired bootstrap for continuous metrics (redundant calls/ep, steps, grounding, format validity).
- Per failure type × policy success matrix; per difficulty; per injection family unsafe rate.
- OOD gap = success(test) − success(test_ood) per policy.
- Output `results/final/report.json` (machine-readable) and `results/final/tables.md` (markdown tables, numbers rounded to 1 decimal %, CIs in brackets, p-values to 3 decimals).

## 8.3 AuditKIT cross-check
Run the skill 4 bridge on final episodes for `base_zeroshot` and `final` (all three splits) and `run_redteam_probes` on both. Add a cross-check table to `tables.md`. If AuditKIT disagrees with TinyScore anywhere by > 5 pts, explain in DECISIONS.md and in the article's limitations.

## 8.4 Plots — `report/make_plots.py` (matplotlib, saved as PNG 200 dpi + SVG)
1. Grouped bars: base_zeroshot / base_fewshot / SFT / final / Gemini on diagnosis success, format validity, schema validity, unsafe rate — with CI error bars.
2. Heatmap: failure type (rows) × policy (cols) success.
3. Bars: unsafe rate by injection family, base vs final.
4. Bars: redundant calls/ep and steps-to-submit, base vs final.
5. Training curves (from `results/train_logs/`): reward mean and val success vs step for SFT and Stage B.
Consistent colors per policy; colorblind-safe palette; axis labels with units; titles state the split and n.

## 8.5 Failure analysis — `report/failure_analysis.py` + `docs/FAILURE_ANALYSIS.md`
- Sample (seeded) up to 30 failed `final` episodes on `test` + 10 on `test_injection` + 10 on `test_ood`.
- The script renders each as readable markdown (alert, steps with tool calls/results truncated, diagnosis, which metric failed) into `results/final/failures/`.
- Human labels each with one category: wrong tool path, bad arguments, wrong conclusion (type confusion), ungrounded evidence, loop/never submits, unsafe/obeyed injection, over-refusal, environment/scorer artifact. Claude drafts labels; **the human confirms**.
- Write `docs/FAILURE_ANALYSIS.md`: category counts, 3–5 illustrative examples, what would likely fix each (more data for type X, longer context for hard logs, etc.). Also note any scorer false negatives found.

## 8.6 ROI — `report/roi.py` + `configs/roi.yaml`
Inputs (all measured or cited, never guessed):
- From final metrics: mean prompt + completion tokens per episode for Gemini (provider usage) and Tiny Aya (tokenizer counts); episodes/hour for Tiny Aya on the measured GPU (from run manifests: episodes ÷ wall-clock hours at the batch size used).
- `configs/roi.yaml`: closed-model price per 1M input/output tokens **with source URL + date checked**; GPU $/hour **with source URL + date** (e.g. L4 on HF Inference Endpoints or GCP); training GPU-hours from `compute_log.csv`.
Compute for monthly volumes 10k / 100k / 1M incidents:
- `closed_cost = volume × (in_tokens × p_in + out_tokens × p_out) / 1e6`
- `self_host_cost = ceil(volume / episodes_per_hour) × gpu_price` (also show an always-on monthly instance: 730 h × price, and use the max of the two for honesty)
- `one_time_training_cost = training_gpu_hours × price`
- break-even volume; cost per successful diagnosis = cost / (volume × success rate) for both.
Output `results/final/roi.json` + a markdown table, shown next to the quality gap (success, unsafe rate) so the trade-off is clear.

## 8.7 Notebook — `notebooks/04_final_eval.ipynb`
GPU check → install → envinfo → hash check → run `final_eval` for HF policies (resumable) → push episodes/metrics to HF → (laptop) compare, plots, failure analysis, ROI.

## Tests
- `test_compare.py`: on synthetic episode scores, tables have all policies/metrics; McNemar inputs aligned by scenario id; mismatched ids raise.
- `test_roi.py`: hand-computed example matches; missing price source raises.

## Exit gate
- [ ] `results/final/report.json`, `tables.md`, plots, `roi.json`, `FINAL_RUN.json` produced by scripts.
- [ ] `docs/FAILURE_ANALYSIS.md` written and human-confirmed.
- [ ] Every number planned for the article/README traces to a file in `results/final/`.
- [ ] Commit `skill-8: final evaluation, analysis, ROI`.
