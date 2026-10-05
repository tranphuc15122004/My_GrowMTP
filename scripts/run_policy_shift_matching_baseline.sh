#!/usr/bin/env bash
# Run from the measured baseline's saved config and initial prepared model.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BASELINE_RUN="${BASELINE_RUN:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3-4b-growmtp-lora/runs/qwen3-4b-growmtp-500steps-20260930T221333Z}"
BASELINE_CONFIG="${BASELINE_CONFIG:-resolved_config-20260930T221455Z-1321745}"
GROWMTP_PYTHON="${GROWMTP_PYTHON:-}"
RUN_MODE="${RUN_MODE:-full}"
TRAIN_STEPS="${TRAIN_STEPS:-}"
GPU_IDS="${GPU_IDS:-0}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

case "$RUN_MODE" in
  smoke)
    TRAIN_STEPS="${TRAIN_STEPS:-4}"
    MODE_OVERRIDES=(
      data.train_batch_size=2 data.max_response_length=128
      actor_rollout_ref.rollout.n=2 actor_rollout_ref.rollout.agent.num_workers=2
      actor_rollout_ref.actor.ppo_mini_batch_size=2
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
      actor_rollout_ref.rollout.max_num_batched_tokens=4096
      actor_rollout_ref.rollout.val_kwargs.n=1 trainer.save_freq=4
    )
    ;;
  full)
    TRAIN_STEPS="${TRAIN_STEPS:-500}"
    MODE_OVERRIDES=()
    ;;
  *) die "RUN_MODE must be 'smoke' or 'full'" ;;
esac

source "$SCRIPT_DIR/python_env.sh"
resolve_growmtp_python "$REPO_ROOT" || die "No Python interpreter found in the active environment or PATH"
source "$SCRIPT_DIR/gpu_args.sh"
parse_gpuid_args "$@" || exit $?
[[ -x "$GROWMTP_PYTHON" ]] || die "Python not found: $GROWMTP_PYTHON"
[[ -f "$BASELINE_RUN/config/$BASELINE_CONFIG.yaml" ]] || die "Saved baseline config not found: $BASELINE_RUN/config/$BASELINE_CONFIG.yaml"
[[ -f "$BASELINE_RUN/prepared_model/config.json" ]] || die "Initial baseline model not found: $BASELINE_RUN/prepared_model"
[[ "$TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]] || die "TRAIN_STEPS must be a positive integer"
if [[ "$RUN_MODE" == smoke ]] && (( TRAIN_STEPS < 4 )); then
  die "Idea smoke needs at least 4 steps to reach the first comparison probe"
fi
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not available; select GPUs on the training server"
GPU_SUMMARY="$(nvidia-smi --id="$GPU_IDS" --query-gpu=index,name --format=csv,noheader 2>/dev/null)" || \
  die "Requested GPU(s) are not available according to nvidia-smi: $GPU_IDS"
GPU_SUMMARY_COUNT="$(printf '%s\n' "$GPU_SUMMARY" | awk 'NF { count++ } END { print count+0 }')"
[[ "$GPU_SUMMARY_COUNT" == "$GPU_COUNT" ]] || die "nvidia-smi found $GPU_SUMMARY_COUNT GPU(s) for GPU_IDS=$GPU_IDS"

BASELINE_RUN="$(realpath -- "$BASELINE_RUN")"
GPU_TAG="${GPU_IDS//,/-}"
IDEA_RUN="${IDEA_RUN:-${BASELINE_RUN%/*}/policy-shift-growmtp-gpu${GPU_TAG}-${TRAIN_STEPS}step-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
IDEA_RUN="$(realpath -m -- "$IDEA_RUN")"
case "$IDEA_RUN" in
    "$BASELINE_RUN"|"$BASELINE_RUN"/*) die "Choose an idea output directory outside the baseline run" ;;
esac
[[ ! -e "$IDEA_RUN" ]] || die "Output already exists; choose a new IDEA_RUN: $IDEA_RUN"
mkdir -p "$IDEA_RUN/logs" "$IDEA_RUN/runtime/tmp"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="$REPO_ROOT/verl:$REPO_ROOT/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
export GROWMTP_RUN_DIR="$IDEA_RUN"
# SGLang uses TMPDIR for Unix-domain IPC sockets, whose paths have a short limit.
# Keep generic temporary files under /tmp and place only the large shift cache on run storage.
export TMPDIR=/tmp
export GROWMTP_SHIFT_CACHE_DIR="$IDEA_RUN/runtime/tmp"

printf 'Python: %s\nBaseline config: %s/config/%s.yaml\n' "$GROWMTP_PYTHON" "$BASELINE_RUN" "$BASELINE_CONFIG"
printf 'Idea run: %s\nMode: %s\nGPU IDs: %s (%s GPUs)\nSteps: %s\n' "$IDEA_RUN" "$RUN_MODE" "$GPU_IDS" "$GPU_COUNT" "$TRAIN_STEPS"
printf 'Metrics: %s/logs/metrics.jsonl\n' "$IDEA_RUN"

exec "$GROWMTP_PYTHON" "$SCRIPT_DIR/run_logged.py" \
  --log-file "$IDEA_RUN/logs/training.log" --level compact -- \
  "$GROWMTP_PYTHON" -m verl.trainer.main_ppo \
  --config-path "$BASELINE_RUN/config" --config-name "$BASELINE_CONFIG" \
  actor_rollout_ref.model.path="$BASELINE_RUN/prepared_model" \
  trainer.resume_mode=disable trainer.default_local_dir="$IDEA_RUN/checkpoints" \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node="$GPU_COUNT" trainer.total_training_steps="$TRAIN_STEPS" \
  ++actor_rollout_ref.model.mtp.comparison_probe_frequency=4 \
  ++actor_rollout_ref.model.mtp.comparison_refresh_fraction=0.25 \
  ++actor_rollout_ref.model.mtp.comparison_probe_max_cycles=4 \
  ++actor_rollout_ref.model.mtp.comparison_probe_max_context=1024 \
  ++actor_rollout_ref.model.mtp.comparison_log_trajectories=true \
  trainer.val_before_train=false trainer.test_freq=-1 ++trainer.final_validation=false \
  trainer.rollout_data_dir="$IDEA_RUN/artifacts/rollouts" \
  ++trainer.validation_data_dir="$IDEA_RUN/artifacts/validation" \
  global_profiler.save_path="$IDEA_RUN/artifacts/profiling" \
  ++ray_kwargs.ray_init._temp_dir="/tmp/gmtp-idea-$$" \
  ++ray_kwargs.ray_init.include_dashboard=false \
  hydra.run.dir="$IDEA_RUN/runtime/hydra" \
  "${MODE_OVERRIDES[@]}"
