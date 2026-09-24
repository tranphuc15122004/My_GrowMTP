#!/usr/bin/env python3
"""Add a missing trainer runtime dependency on top of the cached Modal image."""

from __future__ import annotations

import json
import modal

import scripts.modal_growmtp_smoke_v3 as base

APP = modal.App("growmtp-qwen3-4b-lora-smoke-runtime-preflight")
# Extend the already-built CUDA/Torch/SGLang image by one small CPU-built layer.
IMAGE = base.IMAGE.pip_install("cachetools")

dependency_data_preflight = APP.function(
    image=IMAGE, cpu=4.0, memory=16384, timeout=900, retries=0
)(base.pipeline.dependency_data_preflight.get_raw_f())
run_gpu_smoke = APP.function(
    gpu="H100!", cpu=16.0, memory=131072, timeout=1200, retries=0, image=IMAGE
)(base.pipeline.run_gpu_smoke.get_raw_f())


@APP.local_entrypoint()
def main() -> None:
    print("Running CPU dependency/data preflight first; GPU is not allocated yet.", flush=True)
    preflight = dependency_data_preflight.remote()
    print("CPU preflight passed:", json.dumps(preflight, sort_keys=True), flush=True)
    print("Starting one strict H100 invocation (timeout=20 min, retries=0).", flush=True)
    result = run_gpu_smoke.remote()
    print("FINAL MODAL SMOKE RESULT:", json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
