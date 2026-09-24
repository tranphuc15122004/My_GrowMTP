#!/usr/bin/env python3
"""Corrected Modal entrypoint: copy local source into image before installs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import modal

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_modal_growmtp_smoke as pipeline  # noqa: E402

ROOT = pipeline.ROOT
REMOTE_ROOT = pipeline.REMOTE_ROOT
REMOTE_DATA = pipeline.REMOTE_DATA
TRAIN_NAME = pipeline.TRAIN_NAME
VAL_NAME = pipeline.VAL_NAME

APP = modal.App("growmtp-qwen3-4b-lora-smoke-v2")
IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("build-essential", "cmake", "git", "libnuma-dev", "ninja-build")
    .add_local_dir(str(ROOT / "verl"), remote_path=str(REMOTE_ROOT / "verl"), copy=True)
    .add_local_dir(
        str(ROOT / "sglang/python"),
        remote_path=str(REMOTE_ROOT / "sglang/python"),
        copy=True,
    )
    .add_local_dir(str(ROOT / "scripts"), remote_path=str(REMOTE_ROOT / "scripts"), copy=True)
    .add_local_file(
        str(ROOT / "data" / TRAIN_NAME),
        remote_path=str(REMOTE_DATA / TRAIN_NAME),
        copy=True,
    )
    .add_local_file(
        str(ROOT / "data" / VAL_NAME),
        remote_path=str(REMOTE_DATA / VAL_NAME),
        copy=True,
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install --index-url https://download.pytorch.org/whl/cu130 "
        "torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0",
        f"python {REMOTE_ROOT / 'scripts/install_modal_image_deps.py'} --repo-root {REMOTE_ROOT}",
    )
)

dependency_data_preflight = APP.function(
    image=IMAGE, cpu=4.0, memory=16384, timeout=900, retries=0
)(pipeline.dependency_data_preflight.get_raw_f())
run_gpu_smoke = APP.function(
    gpu="H100!", cpu=16.0, memory=131072, timeout=1200, retries=0, image=IMAGE
)(pipeline.run_gpu_smoke.get_raw_f())


@APP.local_entrypoint()
def main() -> None:
    print("Running CPU dependency/data preflight first; GPU is not allocated yet.", flush=True)
    preflight = dependency_data_preflight.remote()
    print("CPU preflight passed:", json.dumps(preflight, sort_keys=True), flush=True)
    print("Starting one strict H100 invocation (timeout=20 min, retries=0).", flush=True)
    result = run_gpu_smoke.remote()
    print("FINAL MODAL SMOKE RESULT:", json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
