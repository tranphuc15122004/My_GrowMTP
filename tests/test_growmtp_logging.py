import importlib.util
import json
import subprocess
import sys
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
