import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_b200_growmtp_lora.sh"
SERVER_WRAPPER = REPO_ROOT / "scripts" / "run_b200_growmtp_lora_server.sh"


FAKE_PYTHON = r'''#!/usr/bin/env python3
import json
import os
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
    skip_import_preflight=False
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_python = fake_bin / "fake-python"
    fake_python.write_text(FAKE_PYTHON)
    fake_python.chmod(0o755)
    fake_nvidia = fake_bin / "nvidia-smi"
    fake_nvidia.write_text("#!/usr/bin/env bash\nprintf 'NVIDIA B200, 180 GB\\n'\n")
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
    skip_import_preflight=False
):
    return subprocess.run(
        ["bash", str(entrypoint)],
        cwd=REPO_ROOT,
        env=_fake_environment(
            tmp_path, save_generations=save_generations, run_dir=run_dir, log_level=log_level,
            skip_import_preflight=skip_import_preflight
        ),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


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
