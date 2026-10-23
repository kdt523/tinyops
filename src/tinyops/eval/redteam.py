"""Supplementary single-turn red-team check with AuditKIT's RedTeamRunner for one HF policy.

    python -m tinyops.eval.redteam --model CohereLabs/tiny-aya-global --name base_zeroshot
    python -m tinyops.eval.redteam --model CohereLabs/tiny-aya-global --adapter sft+final --name final

The multi-turn `test_injection` split is the primary safety result; this is a cross-check only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tinyops.scoring.auditkit_bridge import run_redteam_probes, single_turn_callable
from tinyops.utils.paths import RESULTS


def run_redteam(policy, out: Path, probes: list[str] | None = None) -> dict:
    """`policy` needs `.act(messages) -> PolicyOutput`."""
    return run_redteam_probes(single_turn_callable(lambda msgs: policy.act(msgs).text), out, probes)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: Annotated[str, typer.Option()],
    name: Annotated[str, typer.Option(help="policy name, used in the output file")],
    adapter: Annotated[str | None, typer.Option()] = None,
    out_dir: Annotated[Path, typer.Option()] = RESULTS / "final" / "auditkit",
    max_new_tokens: Annotated[int, typer.Option()] = 256,
) -> None:
    from tinyops.policies.hf_policy import HFPolicy

    policy = HFPolicy(model, adapter, max_new_tokens=max_new_tokens, name=name)
    report = run_redteam(policy, out_dir / f"redteam_{name}.json")
    typer.echo(report["summary"])


if __name__ == "__main__":
    app()
