import fcntl
import os
import select
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_LOGGED = REPO_ROOT / "scripts" / "run_logged.py"
LAUNCHER = REPO_ROOT / "scripts" / "run_b200_growmtp_lora.sh"


def test_runner_forwards_sigterm_to_execed_training_process(tmp_path):
    stopped_marker = tmp_path / "child-stopped"
    child_script = tmp_path / "train-child.sh"
    child_code = (
        "import pathlib, signal, sys, time\n"
        f"stopped = pathlib.Path({str(stopped_marker)!r})\n"
        "def stop(_signum, _frame):\n"
        "    stopped.write_text('checkpoint-safe-exit')\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "print('GROWMTP_STEP child-ready', flush=True)\n"
        "while True: time.sleep(0.05)\n"
    )
    child_script.write_text(f"#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} -u -c {shlex.quote(child_code)}\n")

    runner = subprocess.Popen(
        [
            sys.executable,
            str(RUN_LOGGED),
            "--log-file",
            str(tmp_path / "training.log"),
            "--level",
            "compact",
            "--",
            "bash",
            str(child_script),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    try:
        ready, _, _ = select.select([runner.stdout], [], [], 5)
        assert ready, "training child did not reach its ready point"
        assert "GROWMTP_STEP child-ready" in runner.stdout.readline()
        runner.send_signal(signal.SIGTERM)
        output, _ = runner.communicate(timeout=5)
        assert runner.returncode == 0
        assert stopped_marker.read_text() == "checkpoint-safe-exit"
        assert "forwarding a graceful signal" in output
    except subprocess.TimeoutExpired:
        pytest.fail("signal did not reach the training process before timeout")
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait(timeout=3)


def test_launcher_rejects_run_directory_with_active_lock(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "config").mkdir()
    (run_dir / "runtime" / "ray").mkdir(parents=True)
    (run_dir / "artifacts" / "profiling").mkdir(parents=True)
    lock_path = run_dir / ".run.lock"

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPO_ROOT,
            env={**os.environ, "RUN_DIR": str(run_dir)},
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

    assert result.returncode != 0
    assert "Another GrowMTP process already holds this run directory" in result.stdout
    assert "nvidia-smi" not in result.stdout


def test_driver_turns_launcher_stop_marker_into_trainer_suspend_request(tmp_path, monkeypatch):
    import importlib
    import sys

    pytest.importorskip("ray")
    pytest.importorskip("omegaconf")
    sys.path.insert(0, str(REPO_ROOT / "verl"))
    main_ppo = importlib.import_module("verl.trainer.main_ppo")
    from omegaconf import OmegaConf

    output_dir = tmp_path / "checkpoints"
    output_dir.mkdir()
    stop_marker = output_dir / ".stop_after_step"
    suspend_marker = output_dir / ".suspend_requested"
    stop_marker.touch()

    class FakeRunMethod:
        def remote(self, _config):
            return "training-ref"

    class FakeRunner:
        run = FakeRunMethod()

    class FakeTaskRunnerClass:
        @staticmethod
        def remote():
            return FakeRunner()

    def wait_for_training(refs, timeout):
        assert refs == ["training-ref"]
        assert timeout == 1.0
        assert suspend_marker.is_file(), "driver did not publish suspension before waiting"
        return refs, []

    monkeypatch.setattr(main_ppo.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(main_ppo.ray, "wait", wait_for_training)
    monkeypatch.setattr(main_ppo.ray, "get", lambda _ref: None)
    config = OmegaConf.create(
        {
            "trainer": {"default_local_dir": str(output_dir)},
            "transfer_queue": {"enable": False},
            "ray_kwargs": {"timeline_json_file": None},
            "global_profiler": {"tool": "none", "steps": []},
        }
    )

    main_ppo.run_ppo(config, task_runner_class=FakeTaskRunnerClass)

    assert suspend_marker.is_file()
    assert stop_marker.is_file()



@pytest.mark.parametrize("log_level", ["compact", "normal", "debug"])
def test_driver_passes_log_level_and_run_root_to_ray_workers(tmp_path, monkeypatch, log_level):
    import importlib
    import sys

    pytest.importorskip("ray")
    pytest.importorskip("omegaconf")
    sys.path.insert(0, str(REPO_ROOT / "verl"))
    main_ppo = importlib.import_module("verl.trainer.main_ppo")
    from omegaconf import OmegaConf

    run_dir = tmp_path / "run"
    output_dir = run_dir / "checkpoints"
    monkeypatch.setenv("GROWMTP_LOG_LEVEL", log_level)
    monkeypatch.setenv("GROWMTP_RUN_DIR", str(run_dir))
    init_kwargs = {}

    class FakeRay:
        @staticmethod
        def init(**kwargs):
            init_kwargs.update(kwargs)

        @staticmethod
        def wait(refs, timeout):
            return refs, []

        @staticmethod
        def get(_ref):
            return None

    class FakeRunMethod:
        def remote(self, _config):
            return "training-ref"

    class FakeRunner:
        run = FakeRunMethod()

    class FakeTaskRunnerClass:
        @staticmethod
        def remote():
            return FakeRunner()

    monkeypatch.setattr(main_ppo.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(main_ppo.ray, "init", FakeRay.init)
    monkeypatch.setattr(main_ppo.ray, "wait", FakeRay.wait)
    monkeypatch.setattr(main_ppo.ray, "get", FakeRay.get)
    monkeypatch.setattr(
        main_ppo,
        "get_ppo_ray_runtime_env",
        lambda: OmegaConf.create({"env_vars": {"DEFAULT_VALUE": "kept"}}),
    )
    config = OmegaConf.create(
        {
            "trainer": {"default_local_dir": str(output_dir)},
            "transfer_queue": {"enable": False},
            "ray_kwargs": {
                "ray_init": {"runtime_env": {"env_vars": {"CUSTOM_VALUE": "kept"}}},
                "timeline_json_file": None,
            },
            "global_profiler": {"tool": "none", "steps": []},
        }
    )

    main_ppo.run_ppo(config, task_runner_class=FakeTaskRunnerClass)

    env_vars = init_kwargs["runtime_env"]["env_vars"]
    assert env_vars["GROWMTP_LOG_LEVEL"] == log_level
    assert env_vars["GROWMTP_RUN_DIR"] == str(run_dir)
    assert env_vars["DEFAULT_VALUE"] == "kept"
    assert env_vars["CUSTOM_VALUE"] == "kept"
    assert list((run_dir / "config").glob("resolved_config-*.yaml"))



def test_repeated_driver_signals_coalesce_into_one_suspend_request(tmp_path, monkeypatch):
    import importlib
    import sys

    pytest.importorskip("ray")
    pytest.importorskip("omegaconf")
    sys.path.insert(0, str(REPO_ROOT / "verl"))
    main_ppo = importlib.import_module("verl.trainer.main_ppo")
    from omegaconf import OmegaConf

    output_dir = tmp_path / "checkpoints"
    output_dir.mkdir()
    suspend_marker = output_dir / ".suspend_requested"
    wait_calls = 0

    class FakeRunMethod:
        def remote(self, _config):
            return "training-ref"

    class FakeRunner:
        run = FakeRunMethod()

    class FakeTaskRunnerClass:
        @staticmethod
        def remote():
            return FakeRunner()

    def wait_for_training(refs, timeout):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            os.kill(os.getpid(), signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGINT)
            return [], refs
        assert suspend_marker.is_file()
        return refs, []

    monkeypatch.setattr(main_ppo.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(main_ppo.ray, "wait", wait_for_training)
    monkeypatch.setattr(main_ppo.ray, "get", lambda _ref: None)
    config = OmegaConf.create(
        {
            "trainer": {"default_local_dir": str(output_dir)},
            "transfer_queue": {"enable": False},
            "ray_kwargs": {"timeline_json_file": None},
            "global_profiler": {"tool": "none", "steps": []},
        }
    )

    main_ppo.run_ppo(config, task_runner_class=FakeTaskRunnerClass)

    assert wait_calls == 2
    assert suspend_marker.is_file()
