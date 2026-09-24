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


MODAL_EXCLUDED_PACKAGES = {
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    # This optional wheel is a 6.3 GiB all-architecture cubin bundle and the
    # updated 0.6.18.post1 build is not published on the public pip index.
    # FlashInfer can fetch only the required, checksummed artifacts lazily.
    "flashinfer-cubin",
}


def modal_runtime_dependencies(raw_dependencies: list[str]) -> list[str]:
    dependencies = []
    for raw in raw_dependencies:
        requirement = Requirement(raw)
        if requirement.name.lower().replace("_", "-") in MODAL_EXCLUDED_PACKAGES:
            continue
        dependencies.append(raw)
    return dependencies


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
    # Preserve the CUDA 13.0 Torch and local-venv Transformers pins. Leave the
    # optional cubin bundle out; FlashInfer downloads artifacts on demand.
    dependencies = modal_runtime_dependencies(project["project"]["dependencies"])

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

    runtime_extras = root / "modal_growmtp_runtime_extras.txt"
    if runtime_extras.is_file():
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "-c",
            str(constraints),
            "-r",
            str(runtime_extras),
        )


if __name__ == "__main__":
    main()
