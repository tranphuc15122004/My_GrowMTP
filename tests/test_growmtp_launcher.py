import json
import os
import runpy
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_b200_growmtp_lora.sh"
SERVER_WRAPPER = REPO_ROOT / "scripts" / "run_b200_growmtp_lora_server.sh"
SUMMARIZATION_WRAPPER = REPO_ROOT / "scripts" / "run_vn_summarization_growmtp.sh"


FAKE_PYTHON = r'''#!/usr/bin/env python3
import json
import os
import runpy
import sys
import time
from pathlib import Path

args = sys.argv[1:]
python_log = os.environ.get("FAKE_PYTHON_LOG")
if python_log:
    with open(python_log, "a") as stream:
        stream.write(json.dumps(args) + "\n")
if args[:3] == ["-m", "verl.trainer.mtp.launch", "prepare"]:
    output = Path(args[args.index("--output") + 1])
    output.mkdir(parents=True, exist_ok=True)
    (output / "prepared.marker").write_text("prepared")
    raise SystemExit(0)
if args[:3] == ["-m", "verl.trainer.mtp.launch", "check"] or args[0:1] == ["-"]:
    raise SystemExit(0)
if args and args[0].endswith("validate_vn_summarization_data.py"):
    sys.argv = args
    runpy.run_path(args[0], run_name="__main__")
    raise SystemExit(0)
if args and args[0].endswith("run_logged.py"):
    assert args[args.index("--level") + 1] == os.environ["GROWMTP_LOG_LEVEL"]
    log_path = Path(args[args.index("--log-file") + 1])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    train_args = args[args.index("--") + 1:]
    output = Path(train_args[train_args.index("--output") + 1])
    run_dir = Path(os.environ["GROWMTP_RUN_DIR"])
    if os.environ.get("FAKE_WAIT_FOR_STOP") != "1":
        (output / ".suspend_requested").unlink(missing_ok=True)
    if os.environ.get("FAKE_WAIT_FOR_STOP") == "1":
        (run_dir / "runtime" / "fake-trainer.pid").write_text(str(os.getpid()))
        print("GROWMTP_STEP | active step 1", flush=True)
        stop_marker = output / ".stop_after_step"
        deadline = time.monotonic() + 10
        while not stop_marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not stop_marker.exists():
            raise SystemExit("launcher did not create stop marker")
        (run_dir / "runtime" / "checkpoint-write-started").touch()
        time.sleep(0.3)
        (output / ".suspend_requested").touch()
    output_actor = output / "global_step_1" / "actor"
    output_actor.mkdir(parents=True, exist_ok=True)
    (output_actor / "fake_checkpoint.marker").write_text("complete")
    if os.environ.get("FAKE_WAIT_FOR_STOP") == "1":
        (output_actor / "safe_stop_done.marker").write_text("step finished")
    (output / "latest_checkpointed_iteration.txt").write_text("1")

    metrics = run_dir / "logs" / "metrics.jsonl"
    with metrics.open("a") as stream:
        stream.write(json.dumps({"step": 1, "actor/pg_loss": 0.25}) + "\n")
    resolved_config = run_dir / "config" / f"resolved_config-{os.getpid()}-{time.time_ns()}.yaml"
    resolved_config.write_text("trainer: fake\n")

    expected_ray = os.environ["RAY_TEMP_DIR"]
    expected_profile = str(run_dir / "artifacts" / "profiling")
    assert expected_ray.startswith("/tmp/gmtp-ray-")
    ray_socket = Path(expected_ray) / "session_2026-09-30_10-48-35_765364_2413816" / "sockets" / "plasma_store"
    assert len(os.fsencode(ray_socket)) <= 107
    assert f"++ray_kwargs.ray_init._temp_dir={expected_ray}" in train_args
    assert f"global_profiler.save_path={expected_profile}" in train_args
    assert "actor_rollout_ref.model.lora_rank=16" in train_args
    assert "actor_rollout_ref.model.lora_alpha=32" in train_args
    assert "actor_rollout_ref.model.lora.merge=true" in train_args
    assert "actor_rollout_ref.model.target_modules=[\"q_proj\",\"v_proj\"]" in train_args
    assert "data.dataloader_num_workers=0" in train_args
    if os.environ.get("FAKE_GCS_FAILURE") == "1":
        session = Path(os.environ["RAY_TEMP_DIR"]) / "session_fake"
        ray_logs = session / "logs"
        ray_logs.mkdir(parents=True)
        (ray_logs / "gcs_server.err").write_text("simulated GCS startup failure\\n")
        (Path(os.environ["RAY_TEMP_DIR"]) / "session_latest").symlink_to(
            session.name, target_is_directory=True
        )
        raise SystemExit(1)
    for arg in train_args:
        if arg.startswith("trainer.rollout_data_dir="):
            Path(arg.split("=", 1)[1]).mkdir(parents=True, exist_ok=True)
        if arg.startswith("trainer.validation_data_dir="):
            Path(arg.split("=", 1)[1]).mkdir(parents=True, exist_ok=True)
    message = "GROWMTP_STEP | 0001/0001 | reward 0.75"
    with log_path.open("a") as stream:
        stream.write(message + "\n")
    print(message, flush=True)
    raise SystemExit(0)
raise SystemExit(0)
'''


def _fake_environment(
    tmp_path, *, save_generations, run_dir=None, wait_for_stop=False, log_level="compact",
    skip_import_preflight=False, fake_ray_failure=False
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_python = fake_bin / "fake-python"
    fake_python.write_text(FAKE_PYTHON)
    fake_python.chmod(0o755)
    fake_nvidia = fake_bin / "nvidia-smi"
    fake_nvidia.write_text(r'''#!/usr/bin/env bash
gpu_ids=0,1
query_index=0
for arg in "$@"; do
    case "$arg" in
        --id=*) gpu_ids="${arg#--id=}" ;;
        --query-gpu=index,*) query_index=1 ;;
    esac
done
IFS=',' read -r -a gpu_ids_array <<< "$gpu_ids"
for gpu_id in "${gpu_ids_array[@]}"; do
    if (( query_index )); then
        printf '%s, ' "$gpu_id"
    fi
    printf 'NVIDIA B200, 180000\n'
done
''')
    fake_nvidia.chmod(0o755)

    inputs = tmp_path / "input"
    inputs.mkdir(exist_ok=True)
    train_file = inputs / "train.parquet"
    validation_file = inputs / "validation.parquet"
    train_file.touch()
    validation_file.touch()

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GROWMTP_PYTHON": str(fake_python),
        "TRAIN_FILE": str(train_file),
        "VAL_FILE": str(validation_file),
        "BASE_MODEL": "shared-model-reference",
        "RUN_BASE_DIR": str(tmp_path / "runs"),
        "RAY_TEMP_DIR": "",
        "SKIP_TRAINING_IMPORT_PREFLIGHT": "1" if skip_import_preflight else "0",
        "FAKE_GCS_FAILURE": "1" if fake_ray_failure else "0",
        "LOG_LEVEL": log_level,
        "SAVE_GENERATIONS": str(save_generations),
        "FAKE_PYTHON_LOG": str(tmp_path / "fake-python.log"),
        "RUN_MODE": "smoke",
        "REQUIRE_B200": "1",
    }
    if run_dir is not None:
        env["RUN_DIR"] = str(run_dir)
    if wait_for_stop:
        env["FAKE_WAIT_FOR_STOP"] = "1"

    return env


def _run_launcher(
    tmp_path, *, save_generations, run_dir=None, entrypoint=LAUNCHER, log_level="compact",
    skip_import_preflight=False, fake_ray_failure=False
):
    return subprocess.run(
        ["bash", str(entrypoint)],
        cwd=REPO_ROOT,
        env=_fake_environment(
            tmp_path, save_generations=save_generations, run_dir=run_dir, log_level=log_level,
            skip_import_preflight=skip_import_preflight,
            fake_ray_failure=fake_ray_failure
        ),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def _training_args(tmp_path):
    calls = [
        json.loads(line)
        for line in (tmp_path / "fake-python.log").read_text().splitlines()
    ]
    training_calls = [call for call in calls if call and call[0].endswith("run_logged.py")]
    return training_calls[-1][training_calls[-1].index("--") + 1:]


@pytest.mark.parametrize(
    ("run_mode", "probe_frequency", "validate", "validation_samples"),
    [("smoke", 0, "false", 1), ("pilot", 0, "false", 1), ("full", 4, "true", 16)],
)
def test_launcher_comparison_measurement_defaults_by_mode(
    tmp_path, run_mode, probe_frequency, validate, validation_samples
):
    env = _fake_environment(tmp_path, save_generations=0)
    env["RUN_MODE"] = run_mode
    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    train_args = _training_args(tmp_path)
    assert f"++actor_rollout_ref.model.mtp.comparison_probe_frequency={probe_frequency}" in train_args
    assert "++actor_rollout_ref.model.mtp.comparison_probe_max_cycles=4" in train_args
    assert "++actor_rollout_ref.model.mtp.comparison_probe_max_context=1024" in train_args
    assert "++actor_rollout_ref.model.mtp.comparison_log_trajectories=true" in train_args
    assert f"++trainer.final_validation={validate}" in train_args
    assert f"trainer.val_before_train={validate}" in train_args
    assert f"actor_rollout_ref.rollout.val_kwargs.n={validation_samples}" in train_args
    assert "trainer.test_freq=-1" in train_args
    assert not any(arg.startswith("trainer.rollout_data_dir=") for arg in train_args)


def test_launcher_aux_ce_disables_exact_kl_probe(tmp_path):
    env = _fake_environment(tmp_path, save_generations=0)
    env.update({
        "RUN_MODE": "full", "MTP_PROBE_FREQ": "7",
        "MTP_AUX_CE_LAMBDA": "0.125", "MTP_AUX_ADVANTAGE_CLIP": "1.5",
    })
    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = set(_training_args(tmp_path))
    assert "++actor_rollout_ref.model.mtp.comparison_probe_frequency=0" in args
    assert "++actor_rollout_ref.model.mtp.rollout_aux_ce_lambda=0.125" in args
    assert "++actor_rollout_ref.model.mtp.rollout_aux_advantage_clip=1.5" in args


def test_launcher_accepts_summarization_reward_and_preserves_it_on_resume(tmp_path):
    env = _fake_environment(tmp_path, save_generations=0)
    reward_file = REPO_ROOT / "scripts" / "vn_summarization_reward.py"
    env.update({
        "GROWMTP_REWARD_FILE": str(reward_file),
        "GROWMTP_ENABLE_THINKING": "0",
        "VAL_MAX_SAMPLES": "32",
    })
    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = set(_training_args(tmp_path))
    assert f"reward.custom_reward_function.path={reward_file}" in args
    assert "++data.apply_chat_template_kwargs.enable_thinking=false" in args
    assert "data.val_max_samples=32" in args
    resume_script = next((tmp_path / "runs").glob("*/config/resume.sh"))
    resume_text = resume_script.read_text()
    assert "GROWMTP_REWARD_FILE" in resume_text
    assert "GROWMTP_ENABLE_THINKING" in resume_text
    assert "VAL_MAX_SAMPLES" in resume_text


def test_launcher_uses_one_seed_for_data_lora_mtp_and_rollout(tmp_path):
    env = _fake_environment(tmp_path, save_generations=0)
    env["GROWMTP_SEED"] = "23"
    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = set(_training_args(tmp_path))
    assert "data.seed=23" in args
    assert "actor_rollout_ref.actor.data_loader_seed=23" in args
    assert "actor_rollout_ref.actor.fsdp_config.seed=23" in args
    assert "actor_rollout_ref.ref.fsdp_config.seed=23" in args
    assert "++actor_rollout_ref.rollout.engine_kwargs.sglang.random_seed=23" in args
    assert "++actor_rollout_ref.model.lora_init_seed=23" in args
    assert "actor_rollout_ref.model.lora_rank=16" in args
    assert "actor_rollout_ref.model.lora_alpha=32" in args
    assert "actor_rollout_ref.model.lora.merge=true" in args

    resume_script = next((tmp_path / "runs").glob("*/config/resume.sh"))
    assert 'export GROWMTP_SEED=23' in resume_script.read_text()


def test_fresh_model_preparation_receives_the_common_seed(tmp_path):
    env = _fake_environment(tmp_path, save_generations=0)
    env["GROWMTP_SEED"] = "31"
    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in (tmp_path / "fake-python.log").read_text().splitlines()]
    prepare_call = next(call for call in calls if call[:3] == ["-m", "verl.trainer.mtp.launch", "prepare"])
    assert prepare_call[prepare_call.index("--seed") + 1] == "31"


def test_summarization_wrapper_sets_task_configuration(tmp_path):
    env = _fake_environment(tmp_path, save_generations=0)
    data_dir = Path(env["TRAIN_FILE"]).parent
    (data_dir / "manifest.json").write_text(json.dumps({
        "seed": 1,
        "train_rows": 8,
        "validation_rows": 2,
        "prompt_audit": {
            "max_prompt_tokens": 3000,
            "max_prompt_length": 4096,
            "tokenizer": "qwen3-4b",
        },
    }))
    env["DATA_DIR"] = str(data_dir)
    result = subprocess.run(
        ["bash", str(SUMMARIZATION_WRAPPER), "--gpuid", "0,1"], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = set(_training_args(tmp_path))
    assert "--response-length" in args
    assert "256" in args
    assert "data.max_prompt_length=4096" in args
    assert "data.train_batch_size=1" in args
    assert "actor_rollout_ref.rollout.n=2" in args
    assert "++actor_rollout_ref.model.mtp.comparison_probe_frequency=0" in args
    assert "++actor_rollout_ref.model.mtp.rollout_aux_ce_lambda=0" in args
    assert "++data.apply_chat_template_kwargs.enable_thinking=false" in args
    assert "data.seed=1" in args
    assert "actor_rollout_ref.model.lora_rank=16" in args


def test_summarization_wrapper_stops_when_manifest_seed_differs(tmp_path):
    env = _fake_environment(tmp_path, save_generations=0)
    data_dir = Path(env["TRAIN_FILE"]).parent
    (data_dir / "manifest.json").write_text(json.dumps({
        "seed": 99,
        "train_rows": 8,
        "validation_rows": 2,
        "prompt_audit": {
            "max_prompt_tokens": 3000,
            "max_prompt_length": 4096,
            "tokenizer": "qwen3-4b",
        },
    }))
    env["DATA_DIR"] = str(data_dir)
    env["GROWMTP_SEED"] = "1"

    result = subprocess.run(
        ["bash", str(SUMMARIZATION_WRAPPER), "--gpuid", "0,1"], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=20, check=False,
    )

    assert result.returncode != 0
    assert "does not match GROWMTP_SEED" in result.stdout + result.stderr
    assert not (next((tmp_path / "runs").glob("*/logs"), tmp_path / "missing") / "training.log").exists()


def test_launcher_saves_and_restores_comparison_measurement_overrides(tmp_path):
    env = _fake_environment(tmp_path, save_generations=0)
    env.update({
        "MTP_PROBE_FREQ": "7", "MTP_PROBE_MAX_CYCLES": "2",
        "MTP_PROBE_MAX_CONTEXT": "384", "COMPARISON_LOG_TRAJECTORIES": "0",
        "FINAL_VALIDATION": "1", "VAL_BEFORE_TRAIN": "1", "VAL_SAMPLES": "8",
    })
    first = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    expected_args = {
        "++actor_rollout_ref.model.mtp.comparison_probe_frequency=7",
        "++actor_rollout_ref.model.mtp.comparison_probe_max_cycles=2",
        "++actor_rollout_ref.model.mtp.comparison_probe_max_context=384",
        "++actor_rollout_ref.model.mtp.comparison_log_trajectories=false",
        "++trainer.final_validation=true", "trainer.val_before_train=true",
        "actor_rollout_ref.rollout.val_kwargs.n=8",
    }
    assert expected_args <= set(_training_args(tmp_path))
    run_dir = next((tmp_path / "runs").iterdir())
    expected_saved = {
        "mtp_probe_freq=7", "mtp_probe_max_cycles=2", "mtp_probe_max_context=384",
        "comparison_log_trajectories=0", "final_validation=1", "val_before_train=1",
        "val_samples=8", "test_freq=-1",
    }
    launch_log = (run_dir / "config" / "launch.txt").read_text().splitlines()
    assert expected_saved <= set(launch_log)

    resumed = subprocess.run(
        ["bash", str(run_dir / "config" / "resume.sh")], cwd=REPO_ROOT,
        env=_fake_environment(tmp_path, save_generations=0, run_dir=run_dir),
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert expected_args <= set(_training_args(tmp_path))
    launch_log = (run_dir / "config" / "launch.txt").read_text().splitlines()
    assert all(launch_log.count(setting) == 2 for setting in expected_saved)


@pytest.mark.parametrize(
    ("setting", "invalid", "message"),
    [
        ("MTP_PROBE_FREQ", "-1", "must be a non-negative integer"),
        ("MTP_PROBE_FREQ", "1.5", "must be a non-negative integer"),
        ("MTP_PROBE_MAX_CYCLES", "0", "must be a positive integer"),
        ("MTP_PROBE_MAX_CONTEXT", "0", "must be a positive integer"),
        ("VAL_SAMPLES", "0", "must be a positive integer"),
        ("FINAL_VALIDATION", "2", "must be 0 or 1"),
        ("VAL_BEFORE_TRAIN", "false", "must be 0 or 1"),
        ("COMPARISON_LOG_TRAJECTORIES", "true", "must be 0 or 1"),
    ],
)
def test_launcher_rejects_invalid_comparison_measurement_settings(
    tmp_path, setting, invalid, message
):
    env = _fake_environment(tmp_path, save_generations=0)
    env[setting] = invalid
    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode != 0
    assert f"{setting} {message}" in result.stdout + result.stderr
    calls = [
        json.loads(line)
        for line in (tmp_path / "fake-python.log").read_text().splitlines()
    ]
    assert not any(call and call[0].endswith("run_logged.py") for call in calls)


def test_launcher_reuses_prepared_model_when_only_hydra_outputs_exist(tmp_path):
    run_dir = tmp_path / "runs" / "previous-ray-failure"
    (run_dir / "prepared_model").mkdir(parents=True)
    (run_dir / "prepared_model" / "prepared.marker").write_text("ready")
    hydra_output = run_dir / "checkpoints" / "outputs" / "2026-09-30" / "12-18-32"
    hydra_output.mkdir(parents=True)
    (hydra_output / "hydra.log").write_text("Ray failed before training")

    result = _run_launcher(tmp_path, save_generations=0, run_dir=run_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Reusing prepared model and restarting training at step 0" in result.stdout
    assert "prepared_model_reused=1" in (run_dir / "config" / "launch.txt").read_text()
    calls = [json.loads(line) for line in (tmp_path / "fake-python.log").read_text().splitlines()]
    assert not any(call[:3] == ["-m", "verl.trainer.mtp.launch", "prepare"] for call in calls)


def test_launcher_copies_ray_logs_when_training_fails(tmp_path):
    result = _run_launcher(
        tmp_path, save_generations=0, fake_ray_failure=True
    )
    assert result.returncode == 1
    run_dir = Path(next((tmp_path / "runs").iterdir()))
    copied_log = next((run_dir / "logs").glob("ray-*/gcs_server.err"))
    assert "simulated GCS startup failure" in copied_log.read_text()
    assert "Ray session logs copied to" in result.stdout


def test_server_wrapper_skips_slow_import_probe_when_requested(tmp_path):
    result = _run_launcher(
        tmp_path,
        save_generations=0,
        entrypoint=SERVER_WRAPPER,
        skip_import_preflight=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [
        json.loads(line)
        for line in (tmp_path / "fake-python.log").read_text().splitlines()
    ]
    assert not any(call and call[0].endswith("check_training_imports.py") for call in calls)
    assert sum(call[:3] == ["-m", "verl.trainer.mtp.launch", "check"] for call in calls) == 1
    assert "Skipping trainer import check" in result.stdout


def test_launcher_isolates_generated_files_and_resumes_in_same_run(tmp_path):
    first = _run_launcher(tmp_path, save_generations=1, entrypoint=SERVER_WRAPPER)
    assert first.returncode == 0, first.stdout + first.stderr
    python_calls = [
        json.loads(line)
        for line in (tmp_path / "fake-python.log").read_text().splitlines()
    ]
    assert sum(call and call[0].endswith("check_training_imports.py") for call in python_calls) == 1
    assert sum(call[:3] == ["-m", "verl.trainer.mtp.launch", "check"] for call in python_calls) == 1
    first_run_dir = Path(next((tmp_path / "runs").iterdir()))

    assert (first_run_dir / "prepared_model" / "prepared.marker").is_file()
    assert (first_run_dir / "checkpoints" / "global_step_1" / "actor" / "fake_checkpoint.marker").is_file()
    assert (first_run_dir / "logs" / "launcher.log").is_file()
    assert "GROWMTP_STEP" in (first_run_dir / "logs" / "training.log").read_text()
    metric = json.loads((first_run_dir / "logs" / "metrics.jsonl").read_text().splitlines()[0])
    assert metric == {"step": 1, "actor/pg_loss": 0.25}
    assert (first_run_dir / "runtime").is_dir()
    assert not (first_run_dir / "runtime" / "ray").exists()
    assert (first_run_dir / "artifacts" / "profiling").is_dir()
    assert (first_run_dir / "artifacts" / "rollouts").is_dir()
    assert (first_run_dir / "artifacts" / "validation").is_dir()
    resolved_configs = list((first_run_dir / "config").glob("resolved_config-*.yaml"))
    assert len(resolved_configs) == 1
    original_config_text = resolved_configs[0].read_text()
    resume_script = first_run_dir / "config" / "resume.sh"
    assert resume_script.is_file()
    assert str(first_run_dir) in resume_script.read_text()
    assert f"export GROWMTP_RESUME_CONFIG={resume_script}" in resume_script.read_text()
    assert "export RAY_TEMP_DIR=/tmp/gmtp-ray-" in resume_script.read_text()
    assert "ray_temp_dir=/tmp/gmtp-ray-" in (first_run_dir / "config" / "launch.txt").read_text()
    assert "result=complete" in (first_run_dir / "config" / "launch.txt").read_text()

    resumed = subprocess.run(
        ["bash", str(first_run_dir / "config" / "resume.sh")],
        cwd=REPO_ROOT,
        env=_fake_environment(tmp_path, save_generations=0, run_dir=first_run_dir),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    launch_log = (first_run_dir / "config" / "launch.txt").read_text()
    assert launch_log.count("run_action=") == 2
    assert "run_action=resume" in launch_log
    assert "result=complete" in launch_log
    assert len((first_run_dir / "logs" / "metrics.jsonl").read_text().splitlines()) == 2
    assert "Action      resume" in resumed.stdout
    assert resolved_configs[0].read_text() == original_config_text
    assert len(list((first_run_dir / "config").glob("resolved_config-*.yaml"))) == 2

    second_fresh = _run_launcher(tmp_path, save_generations=0)
    assert second_fresh.returncode == 0, second_fresh.stdout + second_fresh.stderr
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 2
    assert first_run_dir in run_dirs
    second_run_dir = next(path for path in run_dirs if path != first_run_dir)
    assert second_run_dir != first_run_dir
    assert not (second_run_dir / "artifacts" / "rollouts").exists()
    assert not (second_run_dir / "artifacts" / "validation").exists()



def test_direct_run_dir_resume_requires_saved_resume_script(tmp_path):
    first = _run_launcher(tmp_path, save_generations=0)
    assert first.returncode == 0, first.stdout + first.stderr
    run_dir = Path(next((tmp_path / "runs").iterdir()))
    original_launch_log = (run_dir / "config" / "launch.txt").read_text()

    changed_train_file = tmp_path / "changed-train.parquet"
    changed_train_file.touch()
    env = _fake_environment(tmp_path, save_generations=0, run_dir=run_dir)
    env["TRAIN_FILE"] = str(changed_train_file)
    direct_resume = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert direct_resume.returncode != 0
    assert "config/resume.sh" in direct_resume.stdout + direct_resume.stderr
    assert (run_dir / "config" / "launch.txt").read_text() == original_launch_log


def test_launcher_signal_waits_for_safe_checkpoint_and_reports_resume(tmp_path):
    env = _fake_environment(tmp_path, save_generations=0, wait_for_stop=True)
    runner = subprocess.Popen(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_lines = []
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            readable, _, _ = select.select([runner.stdout], [], [], 0.1)
            if not readable:
                if runner.poll() is not None:
                    break
                continue
            line = runner.stdout.readline()
            output_lines.append(line)
            if "GROWMTP_STEP | active step 1" in line:
                break
        assert any("GROWMTP_STEP | active step 1" in line for line in output_lines), "fake trainer never started"

        runner.send_signal(signal.SIGTERM)
        run_dir = next((tmp_path / "runs").iterdir())
        checkpoint_started = run_dir / "runtime" / "checkpoint-write-started"
        deadline = time.monotonic() + 5
        while not checkpoint_started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert checkpoint_started.is_file(), "trainer did not begin the safe checkpoint"
        runner.send_signal(signal.SIGINT)
        tail, _ = runner.communicate(timeout=15)
        output = "".join(output_lines) + tail
        assert runner.returncode == 0
        checkpoint_dir = run_dir / "checkpoints"
        assert (checkpoint_dir / "global_step_1" / "actor" / "safe_stop_done.marker").read_text() == "step finished"
        assert (checkpoint_dir / "latest_checkpointed_iteration.txt").read_text() == "1"
        assert (checkpoint_dir / ".suspend_requested").is_file()
        assert not (checkpoint_dir / ".stop_after_step").exists()
        assert "Run suspended safely at checkpoint step 1" in output
        assert f"bash {run_dir}/config/resume.sh" in output

        resumed = subprocess.run(
            ["bash", str(run_dir / "config" / "resume.sh")],
            cwd=REPO_ROOT,
            env=_fake_environment(tmp_path, save_generations=0, run_dir=run_dir),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        assert "Resuming GrowMTP checkpoint at step 1" in resumed.stdout
        assert not (checkpoint_dir / ".suspend_requested").exists()
        launch_log = (run_dir / "config" / "launch.txt").read_text()
        assert "run_action=resume" in launch_log
        assert launch_log.rstrip().endswith("result=complete")
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait(timeout=3)
        run_dirs = tmp_path / "runs"
        if run_dirs.exists():
            for run_dir in run_dirs.iterdir():
                pid_file = run_dir / "runtime" / "fake-trainer.pid"
                if pid_file.is_file():
                    try:
                        os.kill(int(pid_file.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass



def test_launcher_accepts_all_documented_log_levels(tmp_path):
    for log_level in ("compact", "normal", "debug"):
        result = _run_launcher(tmp_path, save_generations=0, log_level=log_level)
        assert result.returncode == 0, result.stdout + result.stderr
        run_dir = next(
            path
            for path in (tmp_path / "runs").iterdir()
            if f"log_level={log_level}" in (path / "config" / "launch.txt").read_text()
        )
        assert f"log_level={log_level}" in (run_dir / "config" / "launch.txt").read_text()
        assert f"Console     {log_level}" in result.stdout


def test_launcher_rejects_unknown_log_level_before_gpu_preflight(tmp_path):
    env = _fake_environment(tmp_path, save_generations=0, log_level="verbose")
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode != 0
    assert "LOG_LEVEL must be compact, normal, or debug" in result.stdout
    assert "Visible GPUs" not in result.stdout
