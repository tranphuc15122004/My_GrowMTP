# Qwen3 4B LoRA GrowMTP Run Lifecycle Implementation Plan

> **For agentic workers:** Use the `executing-plans` skill to carry out this plan task by task. Each task has an independent review and validation point.

**Goal:** Make Qwen3 4B LoRA GrowMTP runs resumable and predictable, with graceful suspension, readable logs, and all generated run artifacts saved under one isolated run directory.

**Architecture:** Keep the existing B200 launcher as the owner of the run directory, preflight checks, signal forwarding, and resume command. The veRL trainer observes a stop request at a training step boundary, saves a checkpoint, and exits; the logger retains full output while showing a selected console view and writing scalar metrics as JSONL.

**Tech Stack:** Bash launcher scripts, Python, veRL PPO trainer, Ray, Hydra, PyTorch, pytest.

**Spec:** User requirements in this conversation; operational behavior in `scripts/GROWMTP_B200_RUNBOOK.md`.

## Global Constraints

- Every fresh invocation gets a unique `RUN_DIR`; resume an isolated run only through its generated `config/resume.sh`, which restores the captured launch settings.
- Generated training files belong under that run directory; source model and parquet inputs remain shared read-only inputs.
- SIGINT and SIGTERM request a stop; the trainer finishes the active step, saves a resumable checkpoint, and exits without the launcher force-killing it.
- `LOG_LEVEL` accepts `compact`, `normal`, or `debug`; the complete trainer stream and scalar metrics are saved regardless of console filtering.
- Keep the Qwen3 4B LoRA and GrowMTP training settings unchanged while finishing run lifecycle, logging, and artifact handling.
- Do not claim the B200 training pipeline is validated until a real server smoke run and safe-stop/resume run complete.

## Current Workspace State

The primary implementation is already present in the working tree: launcher layout and lock in `scripts/run_b200_growmtp_lora.sh`, server defaults in `scripts/run_b200_growmtp_lora_server.sh`, signal observation in `verl/verl/trainer/main_ppo.py`, step-boundary checkpointing in `verl/verl/trainer/ppo/ray_trainer.py`, console and JSONL metrics in the logger/tracking modules, and this behavior's user-facing runbook in `scripts/GROWMTP_B200_RUNBOOK.md`. This plan is therefore a review-and-validation plan for the applied changes, with fixes made only where a validation gate exposes a defect.


## Execution Status — 2026-09-28

- The targeted GrowMTP suite passed: 18 tests in the configured training venv.
- Shell syntax, Python compilation, and changed-file whitespace checks passed. The staged `log_growMTP.txt` contains historical trailing whitespace and was preserved unchanged.
- Real B200 smoke, step-boundary checkpoint, and 500-step run remain unverified. This host exposes a Tesla T4 and CUDA driver 12.4; the configured training venv uses PyTorch CUDA 13.0.
- One broad pytest invocation that also collected the legacy Modal smoke module exited with signal 11 during PPO import. The GrowMTP-specific suite passes when run without that unrelated Modal module; investigate the combined native-runtime crash separately if that mixed suite is required.

## File Map

- `scripts/run_b200_growmtp_lora.sh` — fresh run creation, resume selection, locking, signal handling, output routing, resolved launch metadata.
- `scripts/run_b200_growmtp_lora_server.sh` — server-specific model/data defaults and handoff to the common launcher.
- `scripts/run_logged.py` — captures complete trainer stdout/stderr, filters the terminal view, and forwards signals.
- `verl/verl/trainer/main_ppo.py` — converts a signal or `.stop_after_step` marker into `.suspend_requested` while waiting for the Ray task.
- `verl/verl/trainer/ppo/ray_trainer.py` — checks the suspension marker and saves at a safe step boundary.
- `verl/verl/utils/logger/aggregate_logger.py` and `verl/verl/utils/tracking.py` — formats step summaries and appends scalar metrics to JSONL.
- `verl/verl/trainer/config/growmtp/training.yaml`, `verl/verl/trainer/mtp/launch.py`, and `scripts/run_b200_growmtp_lora.sh` — existing LoRA/GrowMTP setup and checkpoint policy; preserve these settings while validating lifecycle behavior.
- `scripts/GROWMTP_B200_RUNBOOK.md` — launch, run-folder, logging, stop, and resume instructions.

### Task 1: Review and lock down the run-directory contract

**Files:**
- Review: `scripts/run_b200_growmtp_lora.sh`
- Review: `scripts/run_b200_growmtp_lora_server.sh`
- Review/update: `scripts/GROWMTP_B200_RUNBOOK.md`

- [x] Confirm fresh mode creates `RUN_BASE_DIR/qwen3-4b-growmtp-<mode>-<UTC timestamp>-<pid>` and places `prepared_model/`, `checkpoints/`, `logs/`, `config/`, `artifacts/`, and `runtime/ray/` beneath it with the fake-trainer launcher integration test.
- [x] Confirm the server wrapper exports its selected model, datasets, log level, and run-directory settings to the common launcher without creating a second run root.
- [x] Confirm a run lock rejects a concurrent launcher using the same directory; a separate run/resume fixture confirms the lock is released when each launcher exits.
- [x] Confirm resume uses the same run directory, prepared model, checkpoint index, and saved launch configuration; direct `RUN_DIR` resume is rejected before metadata changes unless run through `config/resume.sh`.
- [x] Update the runbook to clarify server-wrapper preflight logging and document local tests plus required B200 checks.

**Validation:** The local fake-trainer integration test passed for fresh unique run paths, server-wrapper handoff, optional dumps, resume, and lock contention. A real B200 smoke run remains a server validation gate.

### Task 2: Prove graceful stop, checkpoint, and resume behavior

**Files:**
- Review/update: `scripts/run_b200_growmtp_lora.sh`
- Review/update: `scripts/run_logged.py`
- Review/update: `verl/verl/trainer/main_ppo.py`
- Review/update: `verl/verl/trainer/ppo/ray_trainer.py`
- Add: `tests/test_growmtp_safe_stop.py`

- [x] Add a test for `run_logged.py` that starts the same exec-style child used by `scripts/train.sh`, waits for SIGTERM, and verifies the signal reaches it and its exit status is preserved.
- [ ] Cover the real trainer step-boundary behavior with a B200 smoke run: place `.stop_after_step` after a training step starts, then verify `.suspend_requested` is acted on only after the active step and a complete checkpoint. The local fixture tests only the launcher/driver protocol.
- [ ] In the same B200 validation, request stop before the first step and immediately after a periodic checkpoint; both paths must exit cleanly with `latest_checkpointed_iteration.txt` pointing to a complete checkpoint.
- [x] Verify repeated SIGTERM/SIGINT requests are idempotent and do not interrupt the simulated checkpoint write in the launcher fixture. Repeat this on B200 before signoff.
- [x] Verify the launcher reports the resume script after safe suspension and that the generated script resumes from the saved step in the fake-trainer integration test.

**Validation:** `PYTHONPATH=verl /home/tuantb/fast_infer_text_sum/.venv/bin/python -m pytest -q tests/test_growmtp_launcher.py tests/test_growmtp_logging.py tests/test_growmtp_safe_stop.py` passed 18 tests. Real trainer checkpoint behavior still requires B200 smoke validation.

### Task 3: Verify readable terminal logs and complete saved logs

**Files:**
- Review/update: `scripts/run_logged.py`
- Review/update: `verl/verl/utils/logger/aggregate_logger.py`
- Review/update: `verl/verl/utils/tracking.py`
- Add: `tests/test_growmtp_logging.py`

- [x] Test compact mode with a fake child emitting step metrics, ordinary setup chatter, a warning, and a traceback. The terminal retains the summary and warning/traceback while the full log retains every line.
- [x] Test normal and debug modes show all child output while preserving full-log capture.
- [x] Test metrics JSONL rows contain timestamp/step, finite scalar values or `null`, and omit non-scalar values.
- [x] Test a metrics-file write error produces one warning and does not break logging.
- [x] Confirm `compact`, `normal`, and `debug` are accepted and passed through the server wrapper, launcher, Ray runtime environment, and logger; invalid levels are rejected before GPU preflight.

**Validation:** The targeted test command above passed; the fake launcher test parsed appended JSONL and verified console/full-log behavior. The B200 smoke comparison remains pending.

### Task 4: Verify every generated artifact is run-local and documented

**Files:**
- Review/update: `scripts/run_b200_growmtp_lora.sh`
- Review/update: `scripts/run_b200_growmtp_lora_server.sh`
- Review/update: `verl/verl/trainer/main_ppo.py`
- Review/update: `scripts/GROWMTP_B200_RUNBOOK.md`

- [x] Trace and assert prepared model, checkpoints, launcher/trainer logs, metrics, resolved config snapshots, Ray session, profiler, and optional generation dump paths with the fake launcher integration test.
- [x] Confirm `SAVE_GENERATIONS=0` creates no generation dump directories and `SAVE_GENERATIONS=1` routes dumps below `artifacts/`.
- [x] Confirm resume appends to metrics, preserves earlier resolved config snapshots, and records a new launch result in `config/launch.txt`; the fixture also verifies resumed training output appends to its log.
- [x] Confirm the documented run tree matches actual paths and identifies the shared source model and parquet inputs.

**Validation:** The fake-trainer launcher integration passed with generation saving both disabled and enabled, verified the configured output paths, and exercised resume in the same run directory. A real B200 path audit remains pending with the hardware smoke run.

### Task 5: Run the final validation ladder and record results

**Files:**
- Update: `scripts/GROWMTP_B200_RUNBOOK.md`
- Update: `docs/superpowers/plans/2026-09-28-qwen3-4b-growmtp-lora.md`

- [x] Run shell syntax checks for both launchers and `scripts/run_logged.py`.
- [x] Run Python compilation checks for implementation and test files in the configured training venv.
- [x] Run the targeted launcher, logger, and safe-stop tests: 19 passed in the configured training venv.
- [ ] Complete a real one-step B200 smoke run and verify its checkpoint, config snapshot, full log, compact terminal summary, and metrics JSONL. This host has a Tesla T4, so the launcher requirement cannot be met here.
- [ ] Complete one controlled suspend-and-resume smoke run on B200 and inspect the resumed step and final launcher result.
- [ ] Only after B200 smoke and recovery gates pass, run the configured full 500-step job; inspect the log and checkpoint cadence before treating the pipeline as complete.
- [x] Record local command/results and the required server-only checks in the runbook. B200 checkpoint step/full-run results remain pending because no B200 is available here.

## Acceptance Criteria

- A fresh run has a unique run directory and every generated training artifact is inside it.
- Ctrl-C or SIGTERM results in a checkpointed, cleanly exited run with a working resume command.
- Repeated stop signals do not kill an active checkpoint write or leave the run marked complete when it was suspended.
- Compact output is readable at a glance; full trainer output and per-step metrics remain available on disk.
- The B200 smoke run and suspend/resume run complete successfully before the 500-step run is treated as validated.

## Known Validation Risks

- Ray's `_temp_dir` setting is an internal/unstable option; verify it directs session files on the deployed Ray version.
- The current workspace changes have not yet been proven by a B200 smoke run or suspend/resume run.
- A full training run can be long and write substantial checkpoint data; only start it after the smoke and recovery gates pass.
