# Qwen3 4B LoRA GrowMTP runbook

## Start a run

A launch without `RUN_DIR` creates a timestamped, isolated folder under `RUN_BASE_DIR`. The server wrapper defaults to the 500-step `full` preset and GPUs `0,1`.

```bash
GPU_IDS=0,1 RUN_MODE=full LOG_LEVEL=compact bash scripts/run_b200_growmtp_lora_server.sh
```

Set `GPU_IDS` to the physical GPU indices to use. The launcher checks that each selected card is a B200 with about 180 GB of memory, sets `CUDA_VISIBLE_DEVICES`, and derives `TRAIN_GPUS` from the list. For example, to use four cards:

```bash
GPU_IDS=0,1,2,3 RUN_MODE=full LOG_LEVEL=compact bash scripts/run_b200_growmtp_lora_server.sh
```

Use `RUN_MODE=smoke` for a one-step launch check or `RUN_MODE=pilot` for three steps. The full B200 preset scales the training batch to 8 per GPU, uses rollout `n=4`, caps agent-loop workers at 72, and scales the rollout token budget with GPU count. It defaults to a 1024-token prompt limit, 4096-token response limit, and 0.6 SGLang GPU memory utilization; the latter leaves room for the co-located trainer. Override any setting through its matching environment variable, such as `ROLLOUT_GPU_MEMORY_UTILIZATION=0.65` or `TRAIN_BATCH_SIZE=32`.

After the trainer import check passes once for the same Python environment, set SKIP_TRAINING_IMPORT_PREFLIGHT=1 on a retry. This skips only the import probe; the GrowMTP/SGLang dependency check still runs.

## Run folder

Each run keeps its generated files together:

```text
RUN_DIR/
  prepared_model/          # base model plus the seeded GrowMTP head
  checkpoints/             # resumable trainer state and stop markers
  logs/
    ar_baseline.log        # AR-only SGLang benchmark output
    launcher.log           # preflight and concise launcher output
    ray-*/                 # Ray session logs copied after a trainer failure
    training.log           # timestamped full trainer stdout and stderr
    metrics.jsonl           # scalar training metrics, one record per step
  config/
    launch.txt              # launch settings and each invocation result
    resume.sh               # exact command and environment to resume this run
    resolved_config-*.yaml   # resolved Hydra configuration per trainer start
  artifacts/
    ar_baseline.json       # automatic AR measurement, when enabled
    profiling/              # profiler output when profiling is enabled
    rollouts/               # optional rollout generations
    validation/             # optional validation generations
    comparison/
      trajectories/         # per-step advantage/reward/acceptance JSONL
      probes/               # sampled policy-shift and acceptance-surrogate JSONL
      shifts/               # full trajectory scores and refresh decisions
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

Every scalar metric is also appended to `logs/metrics.jsonl`; complete trainer stdout and stderr go to `logs/training.log` at every level. Before training, the launcher runs a short AR-only SGLang benchmark on the same visible GPUs, prompt source, prompt/response limits, and rollout parallelism. Its detailed result is stored in `artifacts/ar_baseline.json` and its console output in `logs/ar_baseline.log`; the target ms/token baseline is also written as a `phase=ar_baseline` row in `logs/metrics.jsonl`. The baseline is measured from the initial prepared target model once per run and reused after resume. The common launcher writes its preflight checks and status messages to `logs/launcher.log`. The server wrapper runs its initial dependency/import preflight before the run directory is opened, so that initial output appears in the terminal; later launcher output is saved. Logs and metrics append when the run resumes.

GrowMTP runs retain the existing `actor/*` and `actor/mtp/*` metric keys for compatibility and also emit `target/*` aliases for the PPO policy and `draft/*` aliases for the MTP head. Target policy ratio mean/std/min/max, target and draft gradient norms, trainable parameter counts, optimizer step-applied fractions, validation metrics, and target/draft learning rates are available in `metrics.jsonl`. `actor/grad_norm` remains the norm across all trainable model parameters; use `target/grad_norm` and `draft/grad_norm` to inspect each optimizer group separately.

GrowMTP rollout metrics include `draft/acceptance_length`, accepted/proposed draft-token counts, verification-step count, acceptance rate, generated tokens/s/GPU, and milliseconds per generated token. The speedup is `target/ms_per_generated_token ÷ draft/ms_per_generated_token`; the AR numerator is measured automatically before training, while the speculative denominator is measured at every GrowMTP rollout step. To reuse a previously measured scalar baseline, set `AR_BASELINE_MS_PER_TOKEN`; `AR_BASELINE_TOKENS_PER_SECOND_PER_GPU` remains available for compatibility. Set `AR_BASELINE_AUTO=0` to skip automatic measurement; without a supplied baseline, the speedup metric is omitted. `AR_BASELINE_REQUESTS` and `AR_BASELINE_REPEATS` control benchmark size; defaults match the configured rollout request count and use two timed repeats. The automatic AR baseline is anchored to the initial target checkpoint and reused on resume, so the per-step speedup curve compares each GrowMTP step against that same reference.

## Comparison metrics

The full preset records comparison diagnostics and applies selective future-policy draft refresh every four steps, then runs initial/final validation with 16 samples per prompt. Periodic validation remains disabled by default. `MTP_PROBE_FREQ`, `MTP_PROBE_MAX_CYCLES`, `MTP_PROBE_MAX_CONTEXT`, `MTP_REFRESH_FRACTION`, `COMPARISON_LOG_TRAJECTORIES`, `VAL_BEFORE_TRAIN`, `FINAL_VALIDATION`, and `VAL_SAMPLES` control this behavior and are preserved on resume. Smoke/pilot presets disable probing and refresh unless explicitly enabled. See [metric definitions and comparison commands](GROWMTP_COMPARISON_METRICS.md); comparison artifacts do not require `SAVE_GENERATIONS=1`.

Probe cycle/context budgets bound diagnostic samples only. Refresh scores every recorded cycle of positive-advantage trajectories, selects the top fraction across eligible trajectories, and runs one mixed-teacher head update after PPO. Exact full-vocabulary detection uses a temporary FP32 cache on DP rank zero; its disk space and verifier cost scale with recorded positions and vocabulary size. Measure `draft/refresh/time_s`, `draft/shift/cache_bytes`, and E2E before running a long comparison.

## Stop and resume

Press Ctrl-C or send SIGTERM to the launcher to request a safe stop. The launcher records the request; the trainer finishes its current step, writes a checkpoint, and exits. The launcher prints the resume command:

```bash
bash /path/to/RUN_DIR/config/resume.sh
```

That script restores the run's paths, selected physical `GPU_IDS`, GPU count, model and data settings, LoRA options, log level, and Hydra overrides. Make sure the same GPU indices are available on the server. For an isolated run, passing `RUN_DIR` directly cannot resume it; the launcher requires `config/resume.sh` so current shell defaults cannot silently change the saved configuration. A fresh launch without `RUN_DIR` always creates a separate run folder. The run lock prevents a second launcher from using an active `RUN_DIR`.

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
