# Submission checklist

Run `python -m tinyops.report.check_submission` first: it verifies everything below that can be verified by a script and
prints the rest as manual items. Fix every `[FAIL]`, then work through the manual list. Nothing is uploaded or posted
by the tooling unless you pass `--execute` to `python -m tinyops.report.publish push`.

## Order of operations
1. Skills 5–8 done on a GPU; `results/final/FINAL_RUN.json` exists; `docs/FAILURE_ANALYSIS.md` confirmed by you.
2. `python -m tinyops.eval.compare && python -m tinyops.report.make_plots && python -m tinyops.report.roi` (fill the Gemini price from the official page first).
3. `python -m tinyops.report.make_replays` (copies `report.json` into `space/`), `python -m tinyops.report.make_tables --update-readme`.
4. `python -m tinyops.report.cards --user YOUR_HF_USER` and `python -m tinyops.train.model_card --repo-id YOUR_HF_USER/tinyops-tiny-aya-sft`.
5. `python -m tinyops.report.article`, then replace `<github-user>` in README/article/`space/requirements.txt` and `<author>` in `LICENSE`.
6. `python -m tinyops.report.publish plan --user YOUR_HF_USER`, read it, then `push --user YOUR_HF_USER --execute`.
7. Publish the article on Hugging Face, record the demo (`docs/DEMO_SCRIPT.md`), open the issues in `docs/LEXSI_FEEDBACK.md`, submit the form, post.

## Checklist (from the hackathon requirements)
- [ ] Repo public, CI green, README complete
- [ ] Model(s), dataset (with logs + provenance) and Space public on HF; the Space loads in replay mode
- [ ] HF Article published; at least 2 before/after metrics with CIs and the write-up on metric / data / model choices
- [ ] ROI section included (optional but preferred)
- [ ] All links cross-checked (article ↔ Space ↔ model ↔ dataset ↔ GitHub)
- [ ] HF usernames in the submission match the registration form (credits)
- [ ] Submission form filled; confirmation screenshot saved in `docs/`
- [ ] Post on LinkedIn/X tagging Lexsi Labs, Cohere Labs, Hugging Face; share in Discord

## If shortlisted
Priorities: (1) a larger test set for tighter CIs, (2) Aya Expanse 8B as student or teacher, (3) more failure types (network
policy, PVC pending, HPA thrash), (4) live mode on a GPU for Demo Day. Re-run skill 8 for any new model and keep the old numbers.
