"""Build the Space's replay files from the locked final-run episodes (base vs final, same scenarios).

    python -m tinyops.report.make_replays --out space/replays --n 20

Selection rule (stated in the Space README, so the demo is not a highlight reel): scenarios are drawn
from test / test_injection / test_ood into four buckets by outcome, with fixed quotas, seeded:
  A base fails, final succeeds (8)   B both succeed (4)   C both fail (4)   D base succeeds, final fails (4).
If a bucket is short, its quota moves to the next bucket in the order A, B, C, D. The realised counts
are written to `index.json` next to the replays.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Annotated

import typer

from tinyops.eval.score_episodes import load_split_scenarios
from tinyops.scenarios.build_dataset import sub_seed
from tinyops.utils.io import read_jsonl, write_json
from tinyops.utils.paths import DATA, RESULTS, ROOT

FINAL = RESULTS / "final"
QUOTAS = {"A": 8, "B": 4, "C": 4, "D": 4}
BUCKET_NAMES = {"A": "base fails, final succeeds", "B": "both succeed", "C": "both fail",
                "D": "base succeeds, final fails"}
SPLITS = ("test", "test_injection", "test_ood")
SCORE_FIELDS = ("success", "unsafe", "grounding", "keywords_ok", "type_confusion", "n_turns", "termination",
                "over_refusal")


def bucket(base: dict, final: dict) -> str:
    def ok(s: dict) -> bool:
        return s["success"] and not s["unsafe"]

    return {(False, True): "A", (True, True): "B", (False, False): "C", (True, False): "D"}[(ok(base), ok(final))]


def apportion(n: int) -> dict[str, int]:
    """QUOTAS scaled to sum to exactly n (largest remainder), e.g. 20 -> 8/4/4/4."""
    total = sum(QUOTAS.values())
    raw = {k: n * v / total for k, v in QUOTAS.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (raw[k] - out[k], k), reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def curate(rows: dict[str, list[tuple[str, str]]], n: int, seed: int) -> list[tuple[str, str, str]]:
    """rows: bucket -> [(split, scenario_id)]. Returns [(bucket, split, id)] of length <= n."""
    rng = random.Random(sub_seed(seed, "replays"))
    quotas = apportion(n)
    pools = {k: rng.sample(v, len(v)) for k, v in rows.items()}
    chosen: list[tuple[str, str, str]] = []
    spare = 0
    for k in "ABCD":
        take = quotas[k] + spare
        got = pools.get(k, [])[:take]
        spare = take - len(got)
        chosen += [(k, s, i) for s, i in got]
    if spare:  # buckets exhausted: top up from whatever is left, in bucket order
        for k in "ABCD":
            used = {(s, i) for b, s, i in chosen if b == k}
            for s, i in pools.get(k, []):
                if spare and (s, i) not in used:
                    chosen.append((k, s, i))
                    spare -= 1
    return chosen[:n]


def replay_for(sc, base_ep: dict, final_ep: dict, base_score: dict, final_score: dict, bucket_id: str,
               base_label: str, final_label: str) -> dict:
    gt = sc.state.ground_truth
    return {
        "scenario_id": sc.id, "split": sc.split,
        "failure_type": sc.failure_type, "difficulty": sc.difficulty, "alert": sc.alert_text,
        "injected": sc.injected, "injection_family": sc.injection_family,
        "injection_location": sc.injection_location, "injected_payload": gt.injected_payload,
        "bucket": bucket_id, "bucket_name": BUCKET_NAMES[bucket_id],
        "runs": {
            "base": {"label": base_label, "episode": base_ep, "score": {k: base_score[k] for k in SCORE_FIELDS}},
            "final": {"label": final_label, "episode": final_ep, "score": {k: final_score[k] for k in SCORE_FIELDS}},
        },
    }


def build_replays(final_dir: Path = FINAL, out: Path = ROOT / "space" / "replays", n: int = 20, seed: int = 1234,
                  data: Path = DATA, base: str = "base_zeroshot", final: str = "final") -> dict:
    by_bucket: dict[str, list[tuple[str, str]]] = {k: [] for k in "ABCD"}
    loaded: dict[tuple[str, str], tuple[dict, dict, dict, dict]] = {}
    for split in SPLITS:
        bf, ff = final_dir / "metrics" / f"{base}_{split}.scores.jsonl", final_dir / "metrics" / f"{final}_{split}.scores.jsonl"
        if not (bf.exists() and ff.exists()):
            continue
        bs, fs = {r["scenario_id"]: r for r in read_jsonl(bf)}, {r["scenario_id"]: r for r in read_jsonl(ff)}
        be = {r["scenario_id"]: r for r in read_jsonl(final_dir / "episodes" / f"{base}_{split}.jsonl")}
        fe = {r["scenario_id"]: r for r in read_jsonl(final_dir / "episodes" / f"{final}_{split}.jsonl")}
        if set(bs) != set(fs):
            raise ValueError(f"{split}: base and final were not scored on the same scenarios")
        for sid in sorted(bs):
            by_bucket[bucket(bs[sid], fs[sid])].append((split, sid))
            loaded[(split, sid)] = (be[sid], fe[sid], bs[sid], fs[sid])
    picks = curate(by_bucket, n, seed)
    if not picks:
        raise ValueError("no base/final results found: run the final evaluation first")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    scen = {s: load_split_scenarios(s, data) for s in {p[1] for p in picks}}
    index = []
    for b, split, sid in picks:
        be, fe, bs, fs = loaded[(split, sid)]
        rep = replay_for(scen[split][sid], be, fe, bs, fs, b, "Base Tiny Aya", "TinyOps")
        write_json(out / f"{sid}.json", rep)
        index.append({"scenario_id": sid, "split": split, "bucket": b, "injected": rep["injected"],
                      "failure_type": rep["failure_type"]})
    summary = {"n": len(index), "quotas": QUOTAS, "bucket_counts": {k: sum(i["bucket"] == k for i in index) for k in "ABCD"},
               "bucket_sizes_available": {k: len(v) for k, v in by_bucket.items()}, "seed": seed, "replays": index}
    write_json(out / "index.json", summary)
    if (final_dir / "report.json").exists():  # the Space footer reads this copy
        shutil.copyfile(final_dir / "report.json", out.parent / "report.json")
    return summary


app = typer.Typer(add_completion=False)


@app.command()
def main(final_dir: Annotated[Path, typer.Option()] = FINAL, out: Annotated[Path, typer.Option()] = ROOT / "space" / "replays",
         n: Annotated[int, typer.Option()] = 20, seed: Annotated[int, typer.Option()] = 1234) -> None:
    s = build_replays(final_dir, out, n, seed)
    typer.echo(f"{s['n']} replays; bucket counts {s['bucket_counts']} (available {s['bucket_sizes_available']})")


if __name__ == "__main__":
    app()
