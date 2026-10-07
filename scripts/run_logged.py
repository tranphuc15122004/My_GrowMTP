#!/usr/bin/env python3
"""Run a command, save all output, and keep the compact terminal view readable."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


COMPACT_LINE = re.compile(
    r"GROWMTP_STEP|Training Progress|GrowMTP|Qwen3|TaskRunner hostname|Total training steps|"
    r"Size of train dataloader|Resolved training config saved|Preparing |Resuming |Checkpoint|checkpoint|"
    r"suspend|Suspended|complete|Final validation|Initial validation|"
    r"warning|warn:|error|exception|failed|traceback|CUDA devices|Visible GPUs",
    re.IGNORECASE,
)
TRACEBACK_START = re.compile(r"Traceback \(most recent call last\):", re.IGNORECASE)
TRACEBACK_END = re.compile(r"(?:[A-Za-z]+Error|Exception|SystemExit):")


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--level", choices=("compact", "normal", "debug"), default="compact")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main() -> int:
    args = parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["GROWMTP_LOG_LEVEL"] = args.level

    started = time.monotonic()
    with args.log_file.open("a", encoding="utf-8", buffering=1) as logfile:
        log_available = True

        def write_log(line: str) -> None:
            nonlocal log_available
            if not log_available:
                return
            try:
                logfile.write(line)
            except OSError as error:
                log_available = False
                print(f"WARNING: full training log stopped writing to {args.log_file}: {error}", file=sys.stderr)

        write_log(f"\n===== GrowMTP training output started {timestamp()} level={args.level} =====\n")
        process = subprocess.Popen(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=child_env,
        )

        def forward_signal(signum: int, _frame) -> None:
            if process.poll() is None:
                print(
                    "Stop requested; forwarding a graceful signal to the trainer. "
                    "It will checkpoint at the next safe step boundary.",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    process.send_signal(signum)
                except ProcessLookupError:
                    pass

        previous_handlers = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, forward_signal)

        in_traceback = False
        traceback_finished = False
        assert process.stdout is not None
        try:
            for raw_line in process.stdout:
                line = raw_line.rstrip("\r\n")
                write_log(f"{timestamp()} {line}\n")

                if TRACEBACK_START.search(line):
                    in_traceback = True
                    traceback_finished = False
                if args.level != "compact":
                    print(line, flush=True)
                elif in_traceback or COMPACT_LINE.search(line):
                    print(line, flush=True)

                if in_traceback and TRACEBACK_END.search(line):
                    traceback_finished = True
                elif in_traceback and traceback_finished and not line.strip():
                    in_traceback = False
                    traceback_finished = False

            return_code = process.wait()
        finally:
            for signum, previous_handler in previous_handlers.items():
                signal.signal(signum, previous_handler)

        elapsed = time.monotonic() - started
        write_log(
            f"===== GrowMTP training output ended {timestamp()} "
            f"exit={return_code} elapsed_seconds={elapsed:.1f} =====\n"
        )

    if return_code != 0:
        print(f"Trainer exited with status {return_code}; full output is in {args.log_file}", file=sys.stderr)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
