# Qwen3 4B LoRA GrowMTP runbook

## Start a run

A launch without `RUN_DIR` creates a timestamped, isolated folder under `RUN_BASE_DIR`. The B200 launcher defaults to a one-step smoke run; use `RUN_MODE=full` for the 500-step preset.

```bash
RUN_MODE=full LOG_LEVEL=compact bash scripts/run_b200_growmtp_lora.sh
```

On the configured server, use its wrapper to select the server model and validation dataset:

```bash
RUN_MODE=smoke LOG_LEVEL=compact bash scripts/run_b200_growmtp_lora_server.sh
```

After the trainer import check passes once for the same Python environment, set SKIP_TRAINING_IMPORT_PREFLIGHT=1 on a retry. This skips only the import probe; the GrowMTP/SGLang dependency check still runs.

## Run folder

Each run keeps its generated files together:

```text
RUN_DIR/
  prepared_model/          # base model plus the seeded GrowMTP head
  checkpoints/             # resumable trainer state and stop markers
  logs/
    launcher.log           # preflight and concise launcher output
    ray-*/                 # Ray session logs copied after a trainer failure
    training.log           # timestamped full trainer stdout and stderr
    metrics.jsonl           # scalar training metrics, one record per step
  config/
    launch.txt              # launch settings and each invocation result
    resume.sh               # exact command and environment to resume this run
    resolved_config-*.yaml   # resolved Hydra configuration per trainer start
  artifacts/
    profiling/              # profiler output when profiling is enabled
    rollouts/               # optional rollout generations
    validation/             # optional validation generations
  runtime/                  # launcher support state
  .run.lock                 # prevents simultaneous launchers using the same run
```

`SAVE_GENERATIONS=1` enables rollout and validation dumps in the run folder. It is off by default because those dumps can grow quickly. Profiler output is directed into `artifacts/profiling` when profiling is enabled.

The input parquet files and source model remain shared inputs. The prepared model and all training outputs are run-local.

Ray uses a short per-process path under /tmp by default to stay below the Unix socket path limit. Set RAY_TEMP_DIR only if you need to choose another absolute path of at most 39 bytes.

## Console and logs

`LOG_LEVEL` controls the terminal view:

- `compact` (default): one step summary with progress, reward, policy and GrowMTP loss, learning rate, speed, step time, GPU memory, and MTP acceptance.
- `normal`: all trainer output, including the full scalar metric line for each step.
- `debug`: all trainer output plus the complete resolved configuration.

Every scalar metric is also appended to `logs/metrics.jsonl`; complete trainer stdout and stderr go to `logs/training.log` at every level. The common launcher writes its preflight checks and status messages to `logs/launcher.log`. The server wrapper runs its initial dependency/import preflight before the run directory is opened, so that initial output appears in the terminal; later launcher output is saved. Logs and metrics append when the run resumes.

## Stop and resume

Press Ctrl-C or send SIGTERM to the launcher to request a safe stop. The launcher records the request; the trainer finishes its current step, writes a checkpoint, and exits. The launcher prints the resume command:

```bash
bash /path/to/RUN_DIR/config/resume.sh
```

That script restores the run's paths, model and data settings, LoRA options, log level, and Hydra overrides. For an isolated run, passing `RUN_DIR` directly cannot resume it; the launcher requires `config/resume.sh` so current shell defaults cannot silently change the saved configuration. A fresh launch without `RUN_DIR` always creates a separate run folder. The run lock prevents a second launcher from using an active `RUN_DIR`.

## Validation

Run the local launcher, logger, and Ray-driver regression tests in the configured training environment:

```bash
export GROWMTP_PYTHON=/path/to/the/B200/venv/bin/python
export PYTHONPATH="$PWD/verl${PYTHONPATH:+:$PYTHONPATH}"
"$GROWMTP_PYTHON" -m pytest -q \
  tests/test_growmtp_launcher.py \
  tests/test_growmtp_logging.py \
  tests/test_growmtp_safe_stop.py
```

The launcher integration tests stub GPU discovery and the trainer. They verify run-folder routing, stop-marker handling, and resume-script behavior without loading a model. They do not prove the real trainer writes a complete checkpoint at a step boundary.

On a B200 host, run the one-step smoke job, press Ctrl-C after a `GROWMTP_STEP` line appears, wait for the safe-suspension message, and run the printed `config/resume.sh`. Confirm the resumed step, `checkpoints/latest_checkpointed_iteration.txt`, and preserved `config/resolved_config-*.yaml` files. Run the 500-step preset only after this smoke and recovery sequence succeeds.

Current workspace validation record (2026-09-28): the targeted command above completed with 19 passed tests. Shell syntax and Python compilation checks also passed. This host exposes a Tesla T4 with CUDA driver 12.4, while the configured training venv uses PyTorch CUDA 13.0 and the launcher requires a B200; no real training or full 500-step job was launched here.
