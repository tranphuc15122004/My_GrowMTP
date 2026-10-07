import ast
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_LOGGED = REPO_ROOT / "scripts" / "run_logged.py"
LOGGER_PATH = REPO_ROOT / "verl" / "verl" / "utils" / "logger" / "aggregate_logger.py"
LOGGER_SPEC = importlib.util.spec_from_file_location("growmtp_aggregate_logger", LOGGER_PATH)
LOGGER_MODULE = importlib.util.module_from_spec(LOGGER_SPEC)
LOGGER_SPEC.loader.exec_module(LOGGER_MODULE)
LocalLogger = LOGGER_MODULE.LocalLogger


def _run_logged(tmp_path, level, source):
    log_path = tmp_path / "training.log"
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_LOGGED),
            "--log-file",
            str(log_path),
            "--level",
            level,
            "--",
            sys.executable,
            "-u",
            "-c",
            source,
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result, log_path


def test_compact_summary_displays_actual_growmtp_metrics_and_appends_jsonl(tmp_path, capsys):
    metrics_path = tmp_path / "logs" / "metrics.jsonl"
    metrics_path.parent.mkdir()
    logger = LocalLogger(
        log_level="compact",
        total_steps=10,
        metrics_path=metrics_path,
    )

    logger.log(
        {
            "critic/score/mean": 0.75,
            "actor/pg_loss": -0.125,
            "actor/mtp/dca_loss": 0.015,
            "actor/lr": 0.00002,
            "perf/throughput": 120.0,
            "timing_s/step": 12.3,
            "rollout/mtp/acceptance_length": 2.75,
            "actor/perf/max_memory_allocated_gb": 20.4,
            "actor/perf/max_memory_reserved_gb": 25.0,
            "metric_nan": float("nan"),
            "metric_text": "skip this",
            "metric_vector": [1, 2],
            "metric_tensor_vector": LOGGER_MODULE.torch.tensor([1.0, 2.0]),
        },
        step=1,
    )

    output = capsys.readouterr().out
    assert "0001/0010" in output
    assert "reward 0.750" in output
    assert "policy -0.1250" in output
    assert "MTP/DCA 0.015" in output
    assert "MTP accept 2.75" in output
    assert "GPU 20.4 GB/25.0 GB reserved" in output

    row = json.loads(metrics_path.read_text().splitlines()[0])
    assert row["step"] == 1
    assert row["critic/score/mean"] == 0.75
    assert row["metric_nan"] is None
    assert "metric_text" not in row
    assert "metric_vector" not in row
    assert "metric_tensor_vector" not in row
    assert row["timestamp_utc"].endswith("+00:00")


def test_metrics_write_failure_warns_once_and_does_not_break_logging(tmp_path, capsys):
    logger = LocalLogger(
        log_level="normal",
        metrics_path=tmp_path / "missing" / "metrics.jsonl",
    )

    logger.log({"actor/pg_loss": 0.1}, step=1)
    logger.log({"actor/pg_loss": 0.2}, step=2)

    output = capsys.readouterr().out
    assert output.count("WARNING: could not append scalar metrics") == 1
    assert "step:1" in output
    assert "step:2" in output


def test_compact_terminal_filters_chatter_but_training_log_keeps_every_line(tmp_path):
    result, log_path = _run_logged(
        tmp_path,
        "compact",
        "\n".join(
            [
                "print('model shard loaded')",
                "print('GROWMTP_STEP | 0001/0001 | reward 0.75')",
                "print('WARNING: low free memory')",
                "print('Traceback (most recent call last):')",
                "print('  File \\\"train.py\\\", line 1, in <module>')",
                "print('ValueError: invalid batch')",
            ]
        ),
    )

    assert result.returncode == 0
    assert "GROWMTP_STEP | 0001/0001 | reward 0.75" in result.stdout
    assert "WARNING: low free memory" in result.stdout
    assert "Traceback (most recent call last):" in result.stdout
    assert "ValueError: invalid batch" in result.stdout
    assert "model shard loaded" not in result.stdout

    full_log = log_path.read_text()
    for line in (
        "model shard loaded",
        "GROWMTP_STEP | 0001/0001 | reward 0.75",
        "WARNING: low free memory",
        "ValueError: invalid batch",
    ):
        assert line in full_log


@pytest.mark.parametrize("level", ["compact", "normal"])
def test_native_training_progress_shows_eta_after_resume_and_survives_redirection(tmp_path, level):
    # Execute the trainer's actual tqdm construction without importing the GPU trainer.
    trainer = REPO_ROOT / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    tree = ast.parse(trainer.read_text())
    call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "tqdm"
        and any(keyword.arg == "desc" and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "Training Progress" for keyword in node.keywords)
    )
    source = "\n".join([
        "from tqdm import tqdm",
        "import tqdm.std as tqdm_std",
        "from types import SimpleNamespace",
        "clock = [100.0]",
        "tqdm_std.time = lambda: clock[0]",
        "self = SimpleNamespace(total_training_steps=10, global_steps=4)",
        f"bar = {ast.unparse(call)}",
        "clock[0] += 40.0",
        "bar.update(1)",
        "bar.close()",
    ])
    result, log_path = _run_logged(tmp_path, level, source)
    assert result.returncode == 0, result.stderr
    for output in (result.stdout, log_path.read_text()):
        assert "Training Progress" in output
        assert "5/10" in output
        assert "ETA 03:20" in output  # five remaining steps at 40 seconds each
        assert "elapsed 00:40" in output
        assert "40.00s/it" in output
        assert "\r" not in output


def test_native_training_progress_finishes_with_zero_eta(tmp_path):
    trainer = REPO_ROOT / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    tree = ast.parse(trainer.read_text())
    call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "tqdm"
        and any(keyword.arg == "desc" and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "Training Progress" for keyword in node.keywords)
    )
    source = "\n".join([
        "from tqdm import tqdm",
        "import tqdm.std as tqdm_std",
        "from types import SimpleNamespace",
        "clock = [100.0]",
        "tqdm_std.time = lambda: clock[0]",
        "self = SimpleNamespace(total_training_steps=10, global_steps=9)",
        f"bar = {ast.unparse(call)}",
        "clock[0] += 40.0",
        "bar.update(1)",
        "bar.close()",
    ])
    result, log_path = _run_logged(tmp_path, "compact", source)
    assert result.returncode == 0, result.stderr
    for output in (result.stdout, log_path.read_text()):
        assert "10/10" in output
        assert "ETA 00:00" in output


@pytest.mark.parametrize("level", ["normal", "debug"])
def test_verbose_levels_show_all_child_output(tmp_path, level):
    result, log_path = _run_logged(
        tmp_path,
        level,
        "print('ordinary setup detail')\nprint('resolved config: seed=7')",
    )

    assert result.returncode == 0
    assert "ordinary setup detail" in result.stdout
    assert "resolved config: seed=7" in result.stdout
    assert "ordinary setup detail" in log_path.read_text()


def test_runner_preserves_child_exit_status(tmp_path):
    result, log_path = _run_logged(tmp_path, "debug", "print('failed as requested')\nraise SystemExit(7)")

    assert result.returncode == 7
    assert "failed as requested" in result.stdout
    assert "exit=7" in log_path.read_text()



def test_compact_summary_skips_non_scalar_tensor_metric(capsys):
    logger = LocalLogger(log_level="compact")

    logger.log({"actor/pg_loss": LOGGER_MODULE.torch.tensor([0.1, 0.2])}, step=1)

    output = capsys.readouterr().out
    assert "GROWMTP_STEP" in output
    assert "policy" not in output


def _load_tracking(monkeypatch):
    # Keep the real console logger; only substitute unavailable optional imports.
    logger_package = types.ModuleType("verl.utils.logger")
    logger_package.LocalLogger = LocalLogger
    monkeypatch.setitem(sys.modules, "verl.utils.logger", logger_package)
    monkeypatch.setitem(sys.modules, "orjson", types.ModuleType("orjson"))
    path = REPO_ROOT / "verl" / "verl" / "utils" / "tracking.py"
    spec = importlib.util.spec_from_file_location("growmtp_tracking", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module.Tracking


def test_console_tracking_persists_scalars_without_custom_log_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("GROWMTP_LOG_LEVEL", raising=False)
    monkeypatch.delenv("GROWMTP_RUN_DIR", raising=False)
    output_dir = tmp_path / "training_output"
    tracking = _load_tracking(monkeypatch)("test", "run", config={"trainer": {"default_local_dir": str(output_dir)}})

    tracking.log({"comparison/time/step_e2e_s": LOGGER_MODULE.torch.tensor(12.0)}, step=1)

    path = output_dir / "logs" / "metrics.jsonl"
    assert path.is_file()
    assert json.loads(path.read_text())["comparison/time/step_e2e_s"] == 12.0


def test_console_tracking_run_dir_overrides_training_output_and_creates_log_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("GROWMTP_RUN_DIR", str(tmp_path / "explicit_run"))
    monkeypatch.setenv("GROWMTP_LOG_LEVEL", "compact")
    output_dir = tmp_path / "checkpoints"
    tracking = _load_tracking(monkeypatch)("test", "run", config={"trainer": {"default_local_dir": str(output_dir)}})

    tracking.log({"rollout/mtp/acceptance_rate": 0.75}, step=1)

    path = tmp_path / "explicit_run" / "logs" / "metrics.jsonl"
    assert path.is_file()
    assert json.loads(path.read_text())["rollout/mtp/acceptance_rate"] == 0.75
    assert not (output_dir / "logs" / "metrics.jsonl").exists()
