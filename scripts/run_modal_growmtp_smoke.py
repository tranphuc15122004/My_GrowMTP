#!/usr/bin/env python3
"""Run the GrowMTP LoRA smoke pipeline on one Modal H100 (CUDA 13.0)."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/growmtp")
REMOTE_DATA = REMOTE_ROOT / "data"
TRAIN_NAME = "dapo-math-17k-train.parquet"
VAL_NAME = "dapo-math-17k-validation.parquet"
EXPECTED_ROWS = {TRAIN_NAME: 16_398, VAL_NAME: 1_000}
APP = modal.App("growmtp-qwen3-4b-lora-smoke")


def build_smoke_env(
    *,
    python_executable: str,
    train_file: str,
    validation_file: str,
    prepared_model_dir: str,
    run_output_dir: str,
) -> dict[str, str]:
    """Map Modal-mounted paths into the repository's existing B200 runner."""
    env = os.environ.copy()
    env.update(
        {
            "GROWMTP_PYTHON": python_executable,
            "TRAIN_FILE": train_file,
            "VAL_FILE": validation_file,
            "BASE_MODEL": "Qwen/Qwen3-4B",
            "PREPARED_MODEL_DIR": prepared_model_dir,
            "RUN_OUTPUT_DIR": run_output_dir,
            "RUN_MODE": "smoke",
            "REQUIRE_B200": "0",
        }
    )
    return env


def _set_repo_pythonpath(env: dict[str, str]) -> None:
    existing = env.get("PYTHONPATH", "")
    repo_paths = f"{REMOTE_ROOT / 'verl'}:{REMOTE_ROOT / 'sglang/python'}"
    env["PYTHONPATH"] = f"{repo_paths}:{existing}" if existing else repo_paths


def _run(
    command: list[str], *, env: dict[str, str], cwd: Path, label: str
) -> subprocess.CompletedProcess[str]:
    print(f"\n===== {label} =====\n$ {' '.join(command)}", flush=True)
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.monotonic() - started
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if result.stderr:
        print(
            result.stderr,
            end="" if result.stderr.endswith("\n") else "\n",
            file=sys.stderr,
            flush=True,
        )
    print(f"===== {label} exit={result.returncode} elapsed={elapsed:.1f}s =====", flush=True)
    if result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout, stderr=result.stderr
        )
    return result


def _gpu_monitor(stop: threading.Event, samples: list[int]) -> None:
    while not stop.wait(20):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            values = result.stdout.strip().splitlines()[0].split(",")
            used_mib = int(values[0].strip())
            samples.append(used_mib)
            print(
                f"GPU sample: used={used_mib} MiB total={values[1].strip()} MiB "
                f"util={values[2].strip()}%",
                flush=True,
            )


def _verify_parquets() -> dict[str, int]:
    import pyarrow.parquet as pq

    counts = {}
    for name, expected in EXPECTED_ROWS.items():
        path = REMOTE_DATA / name
        parquet = pq.ParquetFile(path)
        actual = parquet.metadata.num_rows
        if actual != expected:
            raise ValueError(f"{path}: expected {expected:,} rows, found {actual:,}")
        columns = set(parquet.schema_arrow.names)
        required = {"prompt", "reward_model", "data_source"}
        if not required.issubset(columns):
            raise ValueError(f"{path}: missing required veRL columns {sorted(required - columns)}")
        counts[name] = actual
        print(f"Data preflight OK: {path} rows={actual:,} columns={sorted(columns)}", flush=True)
    return counts


def _versions() -> dict[str, str]:
    names = (
        "torch",
        "transformers",
        "omegaconf",
        "peft",
        "accelerate",
        "pyarrow",
        "ray",
        "tensordict",
        "torchdata",
        "sglang",
        "sglang-kernel",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "MISSING"
    print("Installed package versions:", json.dumps(versions, sort_keys=True), flush=True)
    return versions


IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("build-essential", "cmake", "git", "libnuma-dev", "ninja-build")
    .add_local_dir(str(ROOT / "verl"), remote_path=str(REMOTE_ROOT / "verl"))
    .add_local_dir(str(ROOT / "sglang/python"), remote_path=str(REMOTE_ROOT / "sglang/python"))
    .add_local_dir(str(ROOT / "scripts"), remote_path=str(REMOTE_ROOT / "scripts"))
    .add_local_file(str(ROOT / "data" / TRAIN_NAME), remote_path=str(REMOTE_DATA / TRAIN_NAME))
    .add_local_file(str(ROOT / "data" / VAL_NAME), remote_path=str(REMOTE_DATA / VAL_NAME))
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install --index-url https://download.pytorch.org/whl/cu130 "
        "torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0",
        f"python {REMOTE_ROOT / 'scripts/install_modal_image_deps.py'} --repo-root {REMOTE_ROOT}",
    )
)


@APP.function(image=IMAGE, cpu=4.0, memory=16384, timeout=900, retries=0)
def dependency_data_preflight() -> dict[str, object]:
    env = os.environ.copy()
    _set_repo_pythonpath(env)
    _versions()
    counts = _verify_parquets()
    _run(
        ["bash", str(REMOTE_ROOT / "scripts/install.sh"), "--check"],
        env=env,
        cwd=REMOTE_ROOT,
        label="GrowMTP repository import check",
    )
    _run(
        [sys.executable, str(REMOTE_ROOT / "scripts/check_training_imports.py")],
        env=env,
        cwd=REMOTE_ROOT,
        label="Actual trainer import preflight (including torchdata)",
    )
    return {"parquet_rows": counts, "package_versions": _versions()}


@APP.function(gpu="H100!", cpu=16.0, memory=131072, timeout=1200, retries=0, image=IMAGE)
def run_gpu_smoke() -> dict[str, object]:
    import torch

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"
    if "H100" not in gpu_name.upper():
        raise RuntimeError(f"Expected strict Modal H100 allocation; got {gpu_name}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"H100 does not report BF16 support (CUDA runtime {torch.version.cuda})")

    print(
        "GPU runtime:",
        json.dumps(
            {
                "device": gpu_name,
                "torch_cuda": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "bf16": torch.cuda.is_bf16_supported(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _run(["nvidia-smi"], env=os.environ.copy(), cwd=REMOTE_ROOT, label="GPU and driver details")
    versions = _versions()
    counts = _verify_parquets()

    with tempfile.TemporaryDirectory(prefix="growmtp-smoke-") as temp_dir:
        temp = Path(temp_dir)
        prepared = temp / "qwen3-4b-growmtp-prepared"
        output = temp / "qwen3-4b-growmtp-lora-run"
        train_file = str(REMOTE_DATA / TRAIN_NAME)
        validation_file = str(REMOTE_DATA / VAL_NAME)
        env = build_smoke_env(
            python_executable=sys.executable,
            train_file=train_file,
            validation_file=validation_file,
            prepared_model_dir=str(prepared),
            run_output_dir=str(output),
        )
        _set_repo_pythonpath(env)

        max_vram_mib: list[int] = []
        stop_monitor = threading.Event()
        monitor = threading.Thread(target=_gpu_monitor, args=(stop_monitor, max_vram_mib), daemon=True)
        monitor.start()
        started = time.monotonic()
        try:
            _run(
                ["bash", str(REMOTE_ROOT / "scripts/run_b200_growmtp_lora.sh")],
                env=env,
                cwd=REMOTE_ROOT,
                label="Fresh Qwen3-4B LoRA + GrowMTP, two training steps",
            )
            checkpoint = output / "global_step_2/actor/huggingface"
            if not checkpoint.is_dir() or not any(checkpoint.iterdir()):
                raise FileNotFoundError(f"Expected non-empty checkpoint at {checkpoint}")

            infer = _run(
                [
                    "bash",
                    str(REMOTE_ROOT / "scripts/infer.sh"),
                    "--model",
                    "qwen3_4b",
                    "--model-path",
                    str(checkpoint),
                    "--depth",
                    "5",
                    "--tp",
                    "1",
                    "--memory-fraction",
                    "0.7",
                    "--eager",
                    "--prompt",
                    "Say exactly: GROWMTP_SMOKE_OK.",
                    "--max-new-tokens",
                    "64",
                ],
                env=env,
                cwd=REMOTE_ROOT,
                label="Load step-2 HF checkpoint with SGLang and generate a short response",
            )
            stdout_lines = [line.strip() for line in infer.stdout.splitlines() if line.strip()]
            if not stdout_lines:
                raise RuntimeError("SGLang returned no stdout text for the short generation")
            try:
                metrics = json.loads(stdout_lines[-1])
            except json.JSONDecodeError as exc:
                raise RuntimeError("SGLang inference did not emit the expected JSON metrics line") from exc
            if not isinstance(metrics, dict) or metrics.get("draft_depth") != 5:
                raise RuntimeError(f"Unexpected SGLang output metrics: {metrics!r}")
            generated_text = "\n".join(stdout_lines[:-1]).strip()
            if not generated_text:
                raise RuntimeError("SGLang completed without non-empty generated text")
            elapsed = time.monotonic() - started
            result = {
                "status": "PASS",
                "gpu": gpu_name,
                "torch_cuda": torch.version.cuda,
                "package_versions": versions,
                "parquet_rows": counts,
                "training_steps": 2,
                "checkpoint": str(checkpoint),
                "generated_text": generated_text,
                "inference_metrics": metrics,
                "elapsed_seconds": round(elapsed, 1),
                "max_observed_vram_mib": max(max_vram_mib, default=None),
            }
            print("SMOKE RESULT:", json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
            return result
        except BaseException:
            print("SMOKE TRACEBACK:\n" + traceback.format_exc(), file=sys.stderr, flush=True)
            raise
        finally:
            stop_monitor.set()
            monitor.join(timeout=5)


@APP.local_entrypoint()
def main() -> None:
    print("Running CPU dependency/data preflight first; GPU is not allocated yet.", flush=True)
    preflight = dependency_data_preflight.remote()
    print("CPU preflight passed:", json.dumps(preflight, sort_keys=True), flush=True)
    print("Starting one strict H100 invocation (timeout=20 min, retries=0).", flush=True)
    result = run_gpu_smoke.remote()
    print("FINAL MODAL SMOKE RESULT:", json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
