"""Record the environment: `python -m tinyops.utils.envinfo` writes results/env.json."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Annotated

import typer

from tinyops.utils.io import write_json
from tinyops.utils.log import get_logger
from tinyops.utils.paths import RESULTS, ROOT

log = get_logger(__name__)

PACKAGES = [
    "agenttune", "auditkit", "trl", "transformers", "torch", "peft", "accelerate", "datasets",
]


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _gpu() -> tuple[bool, str | None, float | None]:
    try:
        import torch
    except ImportError:
        return False, None, None
    if not torch.cuda.is_available():
        return False, None, None
    props = torch.cuda.get_device_properties(0)
    return True, props.name, round(props.total_memory / 1e9, 2)


def collect_env() -> dict:
    cuda, gpu_name, gpu_mem = _gpu()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _version(name) for name in PACKAGES},
        "git_commit": _git_commit(),
        "cuda_available": cuda,
        "gpu_name": gpu_name,
        "gpu_mem_gb": gpu_mem,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def main(
    out: Annotated[Path, typer.Option(help="Where to write env.json")] = RESULTS / "env.json",
) -> None:
    write_json(out, collect_env())
    log.info("wrote %s", out)


if __name__ == "__main__":
    typer.run(main)
