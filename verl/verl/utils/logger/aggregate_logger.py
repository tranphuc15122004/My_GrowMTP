# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A Ray logger will receive logging info from different processes.
"""

import datetime
import json
import logging
import math
import numbers
import pprint

import torch


def concat_dict_to_str(dict: dict, step):
    output = [f"step:{step}"]
    for k, v in dict.items():
        if isinstance(v, numbers.Number):
            output.append(f"{k}:{pprint.pformat(v)}")
    output_str = " - ".join(output)
    return output_str


class LocalLogger:
    """
    A local logger that logs messages to the console.

    Args:
        print_to_console (bool): Whether to print to the console.
    """

    def __init__(self, print_to_console=True, *, log_level="normal", total_steps=None, metrics_path=None):
        self.print_to_console = print_to_console
        self.log_level = log_level
        self.metrics_path = metrics_path
        self.metrics_write_failed = False
        try:
            self.total_steps = int(total_steps) if total_steps is not None else None
        except (TypeError, ValueError):
            self.total_steps = None

    @staticmethod
    def _number(data, *keys):
        for key in keys:
            value = data.get(key)
            if value is None:
                continue
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except (TypeError, ValueError, RuntimeError):
                    continue
            if isinstance(value, numbers.Number):
                return float(value)
        return None

    def _format_compact(self, data, step):
        try:
            current_step = int(step)
        except (TypeError, ValueError):
            current_step = None

        if current_step is not None and self.total_steps:
            progress = min(100.0, 100.0 * current_step / self.total_steps)
            fields = [f"{current_step:04d}/{self.total_steps:04d} · {progress:4.1f}%"]
        else:
            fields = [f"step {step}"]

        metrics = (
            ("reward", ("critic/rewards/mean", "critic/score/mean"), ".3f", ""),
            ("policy", ("target/pg_loss", "actor/pg_loss"), ".4f", ""),
            ("target grad", ("target/grad_norm",), ".3e", ""),
            ("MTP/DCA", ("draft/dca_loss", "actor/mtp/dca_loss"), ".3f", ""),
            ("draft grad", ("draft/grad_norm",), ".3e", ""),
            ("LR", ("target/lr", "actor/lr"), ".2e", ""),
            ("speed", ("perf/throughput",), ".2f", " tok/s"),
            ("step", ("timing_s/step",), ".1f", " s"),
            ("MTP accept", ("draft/acceptance_length", "rollout/mtp/acceptance_length"), ".2f", ""),
            ("MTP speedup", ("draft/speedup_vs_ar",), ".2f", "x"),
        )
        for label, keys, format_spec, suffix in metrics:
            value = self._number(data, *keys)
            if value is not None:
                fields.append(f"{label} {format(value, format_spec)}{suffix}")

        allocated = self._number(data, "actor/perf/max_memory_allocated_gb")
        reserved = self._number(data, "actor/perf/max_memory_reserved_gb")
        if allocated is not None:
            memory = f"GPU {allocated:.1f} GB"
            if reserved is not None:
                memory += f"/{reserved:.1f} GB reserved"
            fields.append(memory)

        return "GROWMTP_STEP │ " + " │ ".join(fields)

    def _write_metrics(self, data, step):
        if self.metrics_path is None:
            return
        row = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "step": int(step) if str(step).isdigit() else step,
        }
        for key, value in data.items():
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except (TypeError, ValueError, RuntimeError):
                    continue
            if isinstance(value, numbers.Real):
                numeric_value = float(value)
                row[key] = numeric_value if math.isfinite(numeric_value) else None
        try:
            with open(self.metrics_path, "a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        except OSError as error:
            if not self.metrics_write_failed:
                print(f"WARNING: could not append scalar metrics to {self.metrics_path}: {error}", flush=True)
                self.metrics_write_failed = True
            self.metrics_path = None

    def flush(self):
        pass

    def log(self, data, step):
        self._write_metrics(data, step)
        if self.print_to_console:
            if self.log_level == "compact":
                output = self._format_compact(data, step)
            else:
                output = concat_dict_to_str(data, step=step)
            print(output, flush=True)


class DecoratorLoggerBase:
    """
    Base class for all decorators that log messages.

    Args:
        role (str): The role (the name) of the logger.
        logger (logging.Logger): The logger instance to use for logging.
        level (int): The logging level.
        rank (int): The rank of the process.
        log_only_rank_0 (bool): If True, only log for rank 0.
    """

    def __init__(
        self, role: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0, log_only_rank_0: bool = True
    ):
        self.role = role
        self.logger = logger
        self.level = level
        self.rank = rank
        self.log_only_rank_0 = log_only_rank_0
        self.logging_function = self.log_by_logging
        if logger is None:
            self.logging_function = self.log_by_print

    def log_by_print(self, log_str):
        if not self.log_only_rank_0 or self.rank == 0:
            print(f"{self.role} {log_str}", flush=True)

    def log_by_logging(self, log_str):
        if self.logger is None:
            raise ValueError("Logger is not initialized")
        if not self.log_only_rank_0 or self.rank == 0:
            self.logger.log(self.level, f"{self.role} {log_str}")


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_with_rank(message: str, rank: int = 0, log_only_rank_0: bool = False):
    """_summary_
    Print a message with rank information.
    This function prints the message only if `log_only_rank_0` is False or if the rank is 0.

    Args:
        message (str): _description_
        rank (int, optional): _description_. Defaults to 0.
        log_only_rank_0 (bool, optional): _description_. Defaults to False.
    """
    if not log_only_rank_0 or rank == 0:
        print(f"[Rank {rank}] {message}", flush=True)


def print_with_rank_and_timer(message: str, rank: int = 0, log_only_rank_0: bool = False):
    """_summary_
    Print a message with rank information and a timestamp.
    This function prints the message only if `log_only_rank_0` is False or if the rank is 0.

    Args:
        message (str): _description_
        rank (int, optional): _description_. Defaults to 0.
        log_only_rank_0 (bool, optional): _description_. Defaults to False.
    """
    now = datetime.datetime.now()
    message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] [Rank {rank}] {message}"
    if not log_only_rank_0 or rank == 0:
        print(message, flush=True)


def log_with_rank(message: str, rank, logger: logging.Logger, level=logging.INFO, log_only_rank_0: bool = False):
    """_summary_
    Log a message with rank information using a logger.
    This function logs the message only if `log_only_rank_0` is False or if the rank is 0.
    Args:
        message (str): The message to log.
        rank (int): The rank of the process.
        logger (logging.Logger): The logger instance to use for logging.
        level (int, optional): The logging level. Defaults to logging.INFO.
        log_only_rank_0 (bool, optional): If True, only log for rank 0. Defaults to False.
    """
    if not log_only_rank_0 or rank == 0:
        logger.log(level, f"[Rank {rank}] {message}")
