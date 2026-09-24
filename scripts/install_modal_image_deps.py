#!/usr/bin/env python3
"""Install repository dependencies into the ephemeral Modal image."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    constraints = root / "scripts/modal_growmtp_constraints.txt"

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-c",
        str(constraints),
        "-r",
        str(root / "verl/requirements.txt"),
    )

    project = tomllib.loads((root / "sglang/python/pyproject.toml").read_text())
    dependencies = []
    for raw in project["project"]["dependencies"]:
        requirement = Requirement(raw)
        if requirement.name.lower().replace("_", "-") in {
            "torch",
            "torchaudio",
            "torchvision",
            "transformers",
        }:
            # Preserve the CUDA 13.0 Torch and local-venv Transformers pins.
            continue
        dependencies.append(raw)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as requirements:
        requirements.write("\n".join(dependencies) + "\n")
        requirements.flush()
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--pre",
            "-c",
            str(constraints),
            "--extra-index-url",
            "https://flashinfer.ai/whl/cu130",
            "-r",
            requirements.name,
        )

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-build-isolation",
        "-e",
        str(root / "sglang/python"),
        "-e",
        str(root / "verl"),
    )


if __name__ == "__main__":
    main()
