#!/usr/bin/env bash
set -Eeuo pipefail

# Each fresh invocation gets a private RUN_DIR. Resume with its generated
# config/resume.sh so the saved settings are restored. LOG_LEVEL controls detail.
# Ctrl-C or SIGTERM asks the trainer to checkpoint at the next safe step boundary.
GROWMTP_PYTHON="${GROWMTP_PYTHON:-}"
DATA_DIR="${DATA_DIR:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/DAPO-math-17.4K}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/test.parquet}"
PROMPT_KEY="${PROMPT_KEY:-prompt}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
RUN_BASE_DIR="${RUN_BASE_DIR:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3-4b-growmtp-lora/runs}"

# smoke: one-step wiring check. pilot: three bounded steps sized per GPU.
# full: 500 steps with batch/workers scaled per GPU; all settings are overridable.
RUN_MODE="${RUN_MODE:-smoke}"
RUN_ACTION="${RUN_ACTION:-auto}"
TRAIN_GPUS_REQUESTED="${TRAIN_GPUS:-}"
TRAIN_GPUS="${TRAIN_GPUS:-2}"
GPU_IDS="${GPU_IDS:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
LOG_LEVEL="${LOG_LEVEL:-compact}"
SAVE_GENERATIONS="${SAVE_GENERATIONS:-0}"
# Bounded distribution probes replay complete prefixes up to this context cap.
MTP_PROBE_MAX_CYCLES="${MTP_PROBE_MAX_CYCLES:-4}"
MTP_PROBE_MAX_CONTEXT="${MTP_PROBE_MAX_CONTEXT:-1024}"
MTP_REFRESH_FRACTION="${MTP_REFRESH_FRACTION:-0.25}"
MTP_AUX_CE_LAMBDA="${MTP_AUX_CE_LAMBDA:-0}"
MTP_AUX_ADVANTAGE_CLIP="${MTP_AUX_ADVANTAGE_CLIP:-2}"
COMPARISON_LOG_TRAJECTORIES="${COMPARISON_LOG_TRAJECTORIES:-1}"
# Keep PEFT training, but merge the adapter into rollout weights because
# SGLang's dynamic-LoRA path rejects GrowMTP's EAGLE speculative decoding.
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
TARGET_MODULES_JSON="${TARGET_MODULES_JSON:-[\"q_proj\",\"v_proj\"]}"

# For full runs, keep this at -1 when VAL_FILE is your held-out test set.
# Set it to e.g. 50 only when VAL_FILE is a separate validation split.
FULL_TEST_FREQ="${FULL_TEST_FREQ:--1}"
REQUIRE_B200="${REQUIRE_B200:-1}"
B200_MIN_MEMORY_MIB="${B200_MIN_MEMORY_MIB:-170000}"
AR_BASELINE_MS_PER_TOKEN="${AR_BASELINE_MS_PER_TOKEN:-}"
AR_BASELINE_TOKENS_PER_SECOND_PER_GPU="${AR_BASELINE_TOKENS_PER_SECOND_PER_GPU:-}"
AR_BASELINE_AUTO="${AR_BASELINE_AUTO:-1}"
AR_BASELINE_REQUESTS="${AR_BASELINE_REQUESTS:-}"
AR_BASELINE_REPEATS="${AR_BASELINE_REPEATS:-2}"
AR_BASELINE_TP_SIZE_PER_REPLICA="${AR_BASELINE_TP_SIZE_PER_REPLICA:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"

# Transformers 5.12.1 with Torch 2.13/CUDA 13 does not have a published
# flash-attn2 kernel variant yet. SDPA is supported on B200 and Blackwell GPUs.
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

# Add repo-specific Hydra overrides here if needed, for example:
# EXTRA_HYDRA_OVERRIDES+=(actor_rollout_ref.actor.optim.lr=1e-6)
EXTRA_HYDRA_OVERRIDES=(
    "++actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}"
)

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "$SCRIPT_DIR/python_env.sh"
resolve_growmtp_python "$REPO_ROOT" || die "No Python interpreter found in the active environment or PATH"
printf 'Using Python: %s\n' "$GROWMTP_PYTHON"
source "$SCRIPT_DIR/gpu_args.sh"
parse_gpuid_args "$@" || exit $?
if [[ -n "$GPU_IDS" ]]; then
    if [[ -n "$TRAIN_GPUS_REQUESTED" && "$TRAIN_GPUS_REQUESTED" != "$GPU_COUNT" ]]; then
        die "TRAIN_GPUS=$TRAIN_GPUS_REQUESTED does not match $GPU_COUNT GPU(s) in GPU_IDS=$GPU_IDS"
    fi
    TRAIN_GPUS="$GPU_COUNT"
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
GROWMTP_ENTRYPOINT="${GROWMTP_ENTRYPOINT:-$SCRIPT_DIR/run_b200_growmtp_lora.sh}"
cd "$REPO_ROOT"

LEGACY_LAYOUT=0
if [[ -n "${RUN_DIR:-}" ]]; then
    if [[ -n "${RUN_OUTPUT_DIR:-}" && -n "${PREPARED_MODEL_DIR:-}" ]]; then
        LEGACY_LAYOUT=1
        CHECKPOINT_DIR="$RUN_OUTPUT_DIR"
    else
        CHECKPOINT_DIR="$RUN_DIR/checkpoints"
        PREPARED_MODEL_DIR="$RUN_DIR/prepared_model"
    fi
elif [[ -n "${RUN_OUTPUT_DIR:-}" && -n "${PREPARED_MODEL_DIR:-}" ]]; then
    # Keep existing runs resumable when callers still provide the old split paths.
    LEGACY_LAYOUT=1
    RUN_DIR="$RUN_OUTPUT_DIR"
    CHECKPOINT_DIR="$RUN_OUTPUT_DIR"
else
    RUN_STAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
    RUN_DIR="$RUN_BASE_DIR/qwen3-4b-growmtp-${RUN_MODE}-${RUN_STAMP}-$$"
    CHECKPOINT_DIR="$RUN_DIR/checkpoints"
    PREPARED_MODEL_DIR="$RUN_DIR/prepared_model"
fi
[[ "$RUN_DIR" == /* ]] || RUN_DIR="$REPO_ROOT/$RUN_DIR"
[[ "$CHECKPOINT_DIR" == /* ]] || CHECKPOINT_DIR="$REPO_ROOT/$CHECKPOINT_DIR"
[[ "$PREPARED_MODEL_DIR" == /* ]] || PREPARED_MODEL_DIR="$REPO_ROOT/$PREPARED_MODEL_DIR"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/gmtp-ray-${UID:-0}-$$}"
[[ "$RAY_TEMP_DIR" == /* ]] || die "RAY_TEMP_DIR must be an absolute path"
RAY_TEMP_DIR_BYTES="$(printf '%s' "$RAY_TEMP_DIR" | LC_ALL=C wc -c)"
(( RAY_TEMP_DIR_BYTES <= 39 )) || die "RAY_TEMP_DIR must be at most 39 bytes to keep Ray Unix socket paths below the system limit"

RUN_DIR_EXISTED=0
[[ ! -e "$RUN_DIR" ]] || RUN_DIR_EXISTED=1
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/config" "$RUN_DIR/runtime" "$RUN_DIR/artifacts/profiling" "$RAY_TEMP_DIR"
export RAY_TEMP_DIR
command -v tee >/dev/null 2>&1 || die "tee is required to save the launcher log"
exec > >(trap '' INT TERM; tee -a "$RUN_DIR/logs/launcher.log") 2>&1
exec {RUN_LOCK_FD}>"$RUN_DIR/.run.lock"
flock -n "$RUN_LOCK_FD" || die "Another GrowMTP process already holds this run directory: $RUN_DIR"
printf 'GrowMTP run folder: %s\nLauncher log: %s\n' "$RUN_DIR" "$RUN_DIR/logs/launcher.log"

TRAIN_PID=""
STOP_REQUESTED=0
forward_graceful_stop() {
    STOP_REQUESTED=1
    if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        mkdir -p "$CHECKPOINT_DIR"
        : > "$CHECKPOINT_DIR/.stop_after_step"
        printf '\nStop requested; the trainer will finish the current step and checkpoint before exiting.\n' >&2
    else
        printf '\nStop requested before the trainer started; training will not be launched.\n' >&2
    fi
}
trap forward_graceful_stop INT TERM

[[ -x "$GROWMTP_PYTHON" ]] || die "Python executable not found: $GROWMTP_PYTHON"
if [[ -n "$AR_BASELINE_MS_PER_TOKEN" && -n "$AR_BASELINE_TOKENS_PER_SECOND_PER_GPU" ]]; then
    die "Set only one of AR_BASELINE_MS_PER_TOKEN or AR_BASELINE_TOKENS_PER_SECOND_PER_GPU"
fi
[[ "$AR_BASELINE_AUTO" == 0 || "$AR_BASELINE_AUTO" == 1 ]] || die "AR_BASELINE_AUTO must be 0 or 1"
if [[ -n "$AR_BASELINE_MS_PER_TOKEN" ]]; then
    [[ "$AR_BASELINE_MS_PER_TOKEN" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
        die "AR_BASELINE_MS_PER_TOKEN must be a positive number"
    "$GROWMTP_PYTHON" -c 'import math, sys; value=float(sys.argv[1]); sys.exit(0 if math.isfinite(value) and value > 0 else 1)' \
        "$AR_BASELINE_MS_PER_TOKEN" || \
        die "AR_BASELINE_MS_PER_TOKEN must be a positive finite number"
    EXTRA_HYDRA_OVERRIDES+=(
        "actor_rollout_ref.model.mtp.ar_baseline_ms_per_token=$AR_BASELINE_MS_PER_TOKEN"
    )
elif [[ -n "$AR_BASELINE_TOKENS_PER_SECOND_PER_GPU" ]]; then
    [[ "$AR_BASELINE_TOKENS_PER_SECOND_PER_GPU" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
        die "AR_BASELINE_TOKENS_PER_SECOND_PER_GPU must be a positive number"
    "$GROWMTP_PYTHON" -c 'import math, sys; value=float(sys.argv[1]); sys.exit(0 if math.isfinite(value) and value > 0 else 1)' \
        "$AR_BASELINE_TOKENS_PER_SECOND_PER_GPU" || \
        die "AR_BASELINE_TOKENS_PER_SECOND_PER_GPU must be a positive finite number"
    EXTRA_HYDRA_OVERRIDES+=(
        "actor_rollout_ref.model.mtp.ar_baseline_tokens_per_second_per_gpu=$AR_BASELINE_TOKENS_PER_SECOND_PER_GPU"
    )
fi
[[ -f "$TRAIN_FILE" ]] || die "Training parquet not found: $TRAIN_FILE"
[[ -f "$VAL_FILE" ]] || die "Validation/test parquet not found: $VAL_FILE"
[[ "$LORA_RANK" =~ ^[1-9][0-9]*$ ]] || die "LORA_RANK must be a positive integer"
[[ "$LORA_ALPHA" =~ ^[1-9][0-9]*$ ]] || die "LORA_ALPHA must be a positive integer"
[[ "$TRAIN_GPUS" =~ ^[1-9][0-9]*$ ]] || die "TRAIN_GPUS must be a positive integer"
[[ "$DATALOADER_NUM_WORKERS" =~ ^[0-9]+$ ]] || die "DATALOADER_NUM_WORKERS must be a non-negative integer"
[[ "$SAVE_GENERATIONS" == 0 || "$SAVE_GENERATIONS" == 1 ]] || die "SAVE_GENERATIONS must be 0 or 1"
case "$LOG_LEVEL" in
    compact|normal|debug) ;;
    *) die "LOG_LEVEL must be compact, normal, or debug" ;;
esac
command -v setsid >/dev/null 2>&1 || die "setsid is required for safe signal forwarding"
command -v flock >/dev/null 2>&1 || die "flock is required to prevent overlapping runs in one RUN_DIR"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not available; run this script on the GPU server"
if [[ -n "$GPU_IDS" ]]; then
    GPU_SUMMARY="$(nvidia-smi --id="$GPU_IDS" --query-gpu=index,name,memory.total --format=csv,noheader)"
else
    GPU_SUMMARY="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
fi
printf 'Visible GPUs:\n%s\n' "$GPU_SUMMARY"

export REQUIRE_B200 TRAIN_GPUS
"$GROWMTP_PYTHON" - <<'PY'
import os
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the selected Python environment")
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print("PyTorch CUDA devices:", names)
expected = int(os.environ["TRAIN_GPUS"])
if len(names) != expected:
    raise SystemExit(
        f"TRAIN_GPUS={expected}, but PyTorch sees {len(names)} GPU(s); "
        "select the intended cards with CUDA_VISIBLE_DEVICES"
    )
if not torch.cuda.is_bf16_supported():
    raise SystemExit("This GrowMTP setup expects a GPU with BF16 support")
if os.environ.get("REQUIRE_B200", "1") == "1" and not any("B200" in name.upper() for name in names):
    raise SystemExit("No B200 is visible to this Python process; refusing to launch")
PY

if [[ "${GROWMTP_PREFLIGHT_PASSED:-0}" != "1" ]]; then
    printf '\nChecking GrowMTP dependencies with %s\n' "$GROWMTP_PYTHON"
    GROWMTP_PYTHON="$GROWMTP_PYTHON" bash "$REPO_ROOT/scripts/install.sh" --check
fi

"$GROWMTP_PYTHON" - "$TRAIN_FILE" "$VAL_FILE" "$PROMPT_KEY" <<'PY'
import sys
import pyarrow.parquet as pq

prompt_key = sys.argv[3]
required = {prompt_key, "reward_model", "data_source"}
for path in sys.argv[1:3]:
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema_arrow.names)
    missing = required - columns
    if missing:
        raise SystemExit(
            f"{path}: missing veRL columns {sorted(missing)}; "
            "convert raw problem/answer data with scripts/prepare_data.sh first"
        )
    if parquet.metadata.num_rows < 1 or parquet.metadata.num_row_groups < 1:
        raise SystemExit(f"{path}: parquet has no rows")
    sample = parquet.read_row_group(
        0, columns=[prompt_key, "reward_model", "data_source"]
    ).slice(0, 1).to_pylist()[0]
    prompt = sample[prompt_key]
    reward = sample["reward_model"]
    if not isinstance(prompt, list) or not prompt or not prompt[0].get("content"):
        raise SystemExit(f"{path}: prompt must be a non-empty chat-message list")
    if not isinstance(reward, dict) or not reward.get("ground_truth"):
        raise SystemExit(f"{path}: reward_model.ground_truth is missing")
    print(f"Dataset OK: {path} ({parquet.metadata.num_rows:,} rows)")
PY

PREPARED_MODEL_REUSE=0
if [[ "$RUN_ACTION" == "auto" ]]; then
    if [[ -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]]; then
        RUN_ACTION="resume"
    elif (( RUN_DIR_EXISTED )) && [[ -d "$PREPARED_MODEL_DIR" ]] && \
        [[ ! -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]] && \
        [[ -z "$(find "$CHECKPOINT_DIR" -type d -name 'global_step_*' -print -quit 2>/dev/null || true)" ]]; then
        # Ignore Hydra's outputs/ metadata; only trainer checkpoint markers block model reuse.
        RUN_ACTION="fresh"
        PREPARED_MODEL_REUSE=1
    else
        RUN_ACTION="fresh"
    fi
fi

if [[ "$RUN_ACTION" == "resume" && ( "$LEGACY_LAYOUT" == 0 || -f "$RUN_DIR/config/resume.sh" ) && "${GROWMTP_RESUME_CONFIG:-}" != "$RUN_DIR/config/resume.sh" ]]; then
    die "This run has saved launch settings; resume with: bash $RUN_DIR/config/resume.sh"
fi

case "$RUN_ACTION" in
    fresh)
        if (( RUN_DIR_EXISTED && ! PREPARED_MODEL_REUSE )); then
            die "Run directory already exists; set a new RUN_DIR for a fresh run: $RUN_DIR"
        fi
        RESUME_MODE=disable
        ;;
    resume)
        [[ -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]] || die "No checkpoint index found to resume: $CHECKPOINT_DIR"
        CHECKPOINT_STEP="$(< "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt")"
        [[ "$CHECKPOINT_STEP" =~ ^[0-9]+$ ]] || die "Invalid checkpoint index in $CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
        [[ -d "$CHECKPOINT_DIR/global_step_$CHECKPOINT_STEP/actor" ]] || die "Checkpoint files are missing for step $CHECKPOINT_STEP"
        [[ -d "$PREPARED_MODEL_DIR" ]] || die "Prepared model not found for resume: $PREPARED_MODEL_DIR"
        RESUME_MODE=auto
        ;;
    *)
        die "RUN_ACTION must be 'auto', 'fresh', or 'resume'"
        ;;
esac

case "$RUN_MODE" in
    smoke)
        TRAIN_STEPS="${TRAIN_STEPS:-1}"
        RESPONSE_LENGTH="${RESPONSE_LENGTH:-128}"
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
        ROLLOUT_N="${ROLLOUT_N:-2}"
        AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-$ROLLOUT_N}"
        PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
        MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
        MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-8192}"
        SAVE_FREQ="${SAVE_FREQ:-1}"
        TEST_FREQ="${TEST_FREQ:--1}"
        MTP_PROBE_FREQ="${MTP_PROBE_FREQ:-0}"
        FINAL_VALIDATION="${FINAL_VALIDATION:-0}"
        VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-0}"
        VAL_SAMPLES="${VAL_SAMPLES:-1}"
        ;;
    pilot)
        TRAIN_STEPS="${TRAIN_STEPS:-3}"
        RESPONSE_LENGTH="${RESPONSE_LENGTH:-1024}"
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((2 * TRAIN_GPUS))}"
        ROLLOUT_N="${ROLLOUT_N:-2}"
        AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-$TRAIN_GPUS}"
        PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
        MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-$((4096 * TRAIN_GPUS))}"
        MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-8192}"
        SAVE_FREQ="${SAVE_FREQ:-1}"
        TEST_FREQ="${TEST_FREQ:--1}"
        MTP_PROBE_FREQ="${MTP_PROBE_FREQ:-0}"
        FINAL_VALIDATION="${FINAL_VALIDATION:-0}"
        VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-0}"
        VAL_SAMPLES="${VAL_SAMPLES:-1}"
        ;;
    full)
        TRAIN_STEPS="${TRAIN_STEPS:-500}"
        RESPONSE_LENGTH="${RESPONSE_LENGTH:-4096}"
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((8 * TRAIN_GPUS))}"
        ROLLOUT_N="${ROLLOUT_N:-4}"
        DEFAULT_AGENT_LOOP_WORKERS=$((4 * TRAIN_GPUS))
        (( DEFAULT_AGENT_LOOP_WORKERS <= 72 )) || DEFAULT_AGENT_LOOP_WORKERS=72
        AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-$DEFAULT_AGENT_LOOP_WORKERS}"
        PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
        MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-$((8192 * TRAIN_GPUS))}"
        MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-16384}"
        SAVE_FREQ="${SAVE_FREQ:-10}"
        TEST_FREQ="${TEST_FREQ:-$FULL_TEST_FREQ}"
        MTP_PROBE_FREQ="${MTP_PROBE_FREQ:-4}"
        FINAL_VALIDATION="${FINAL_VALIDATION:-1}"
        VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-1}"
        VAL_SAMPLES="${VAL_SAMPLES:-16}"
        ;;
    *)
        die "RUN_MODE must be 'smoke', 'pilot', or 'full'"
        ;;
esac

if [[ -z "$AR_BASELINE_REQUESTS" ]]; then
    AR_BASELINE_REQUESTS=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
fi

for setting in TRAIN_STEPS RESPONSE_LENGTH TRAIN_BATCH_SIZE ROLLOUT_N AGENT_LOOP_WORKERS \
    PPO_MINI_BATCH_SIZE MAX_BATCHED_TOKENS MAX_TOKEN_LEN_PER_GPU SAVE_FREQ MAX_PROMPT_LENGTH \
    AR_BASELINE_REQUESTS AR_BASELINE_REPEATS AR_BASELINE_TP_SIZE_PER_REPLICA \
    MTP_PROBE_MAX_CYCLES MTP_PROBE_MAX_CONTEXT VAL_SAMPLES; do
    [[ "${!setting}" =~ ^[1-9][0-9]*$ ]] || die "$setting must be a positive integer"
done
[[ "$MTP_PROBE_FREQ" =~ ^[0-9]+$ ]] || die "MTP_PROBE_FREQ must be a non-negative integer"
for setting in FINAL_VALIDATION VAL_BEFORE_TRAIN COMPARISON_LOG_TRAJECTORIES; do
    [[ "${!setting}" == 0 || "${!setting}" == 1 ]] || die "$setting must be 0 or 1"
done
FINAL_VALIDATION_HYDRA=false
[[ "$FINAL_VALIDATION" == 0 ]] || FINAL_VALIDATION_HYDRA=true
VAL_BEFORE_TRAIN_HYDRA=false
[[ "$VAL_BEFORE_TRAIN" == 0 ]] || VAL_BEFORE_TRAIN_HYDRA=true
COMPARISON_LOG_TRAJECTORIES_HYDRA=false
[[ "$COMPARISON_LOG_TRAJECTORIES" == 0 ]] || COMPARISON_LOG_TRAJECTORIES_HYDRA=true
(( TRAIN_GPUS % AR_BASELINE_TP_SIZE_PER_REPLICA == 0 )) || \
    die "TRAIN_GPUS must be divisible by AR_BASELINE_TP_SIZE_PER_REPLICA"
"$GROWMTP_PYTHON" -c 'import math, sys; value=float(sys.argv[1]); sys.exit(0 if math.isfinite(value) and 0 < value < 1 else 1)' \
    "$ROLLOUT_GPU_MEMORY_UTILIZATION" || die "ROLLOUT_GPU_MEMORY_UTILIZATION must be between 0 and 1"
"$GROWMTP_PYTHON" -c 'import math, sys; value=float(sys.argv[1]); sys.exit(0 if math.isfinite(value) and 0 <= value <= 1 else 1)' \
    "$MTP_REFRESH_FRACTION" || die "MTP_REFRESH_FRACTION must be between 0 and 1"
"$GROWMTP_PYTHON" -c 'import math, sys; value=float(sys.argv[1]); sys.exit(0 if math.isfinite(value) and value >= 0 else 1)' \
    "$MTP_AUX_CE_LAMBDA" || die "MTP_AUX_CE_LAMBDA must be a finite nonnegative number"
"$GROWMTP_PYTHON" -c 'import math, sys; value=float(sys.argv[1]); sys.exit(0 if math.isfinite(value) and value > 0 else 1)' \
    "$MTP_AUX_ADVANTAGE_CLIP" || die "MTP_AUX_ADVANTAGE_CLIP must be a finite positive number"
if awk -v value="$MTP_AUX_CE_LAMBDA" 'BEGIN { exit !(value + 0 > 0) }'; then
    # The rollout-advantage objective is piloted separately from exact-KL refresh.
    MTP_PROBE_FREQ=0
fi
[[ "$TEST_FREQ" == "-1" || "$TEST_FREQ" =~ ^[1-9][0-9]*$ ]] || \
    die "TEST_FREQ must be -1 or a positive integer"
(( PPO_MINI_BATCH_SIZE <= TRAIN_BATCH_SIZE * ROLLOUT_N )) || \
    die "PPO_MINI_BATCH_SIZE must not exceed TRAIN_BATCH_SIZE * ROLLOUT_N"
(( MAX_TOKEN_LEN_PER_GPU >= RESPONSE_LENGTH )) || \
    die "MAX_TOKEN_LEN_PER_GPU must be at least RESPONSE_LENGTH"

write_resume_config() {
    local resume_script="$RUN_DIR/config/resume.sh"
    {
        printf '#!/usr/bin/env bash\nset -euo pipefail\n'
        local name
        for name in \
            GROWMTP_PYTHON DATA_DIR TRAIN_FILE VAL_FILE PROMPT_KEY BASE_MODEL RUN_BASE_DIR RUN_DIR RAY_TEMP_DIR \
            RUN_MODE GPU_IDS TRAIN_GPUS B200_MIN_MEMORY_MIB LOG_LEVEL SAVE_GENERATIONS DATALOADER_NUM_WORKERS LORA_RANK LORA_ALPHA \
            TARGET_MODULES_JSON FULL_TEST_FREQ REQUIRE_B200 ATTN_IMPLEMENTATION TRAIN_STEPS \
            RESPONSE_LENGTH TRAIN_BATCH_SIZE ROLLOUT_N AGENT_LOOP_WORKERS PPO_MINI_BATCH_SIZE \
            MAX_BATCHED_TOKENS MAX_TOKEN_LEN_PER_GPU SAVE_FREQ TEST_FREQ \
            MTP_PROBE_FREQ MTP_PROBE_MAX_CYCLES MTP_PROBE_MAX_CONTEXT COMPARISON_LOG_TRAJECTORIES \
            MTP_REFRESH_FRACTION MTP_AUX_CE_LAMBDA MTP_AUX_ADVANTAGE_CLIP \
            FINAL_VALIDATION VAL_BEFORE_TRAIN VAL_SAMPLES \
            MAX_PROMPT_LENGTH ROLLOUT_GPU_MEMORY_UTILIZATION \
            AR_BASELINE_MS_PER_TOKEN AR_BASELINE_TOKENS_PER_SECOND_PER_GPU AR_BASELINE_AUTO \
            AR_BASELINE_REQUESTS AR_BASELINE_REPEATS AR_BASELINE_TP_SIZE_PER_REPLICA \
            GROWMTP_ENTRYPOINT; do
            printf 'export %s=%q\n' "$name" "${!name}"
        done
        if (( LEGACY_LAYOUT )); then
            printf 'export PREPARED_MODEL_DIR=%q\n' "$PREPARED_MODEL_DIR"
            printf 'export RUN_OUTPUT_DIR=%q\n' "$RUN_OUTPUT_DIR"
        fi
        printf 'export GROWMTP_RESUME_CONFIG=%q\n' "$resume_script"
        printf 'export RUN_ACTION=auto\n'
        printf 'EXTRA_HYDRA_OVERRIDES=('
        printf ' %q' "${EXTRA_HYDRA_OVERRIDES[@]}"
        printf ' )\nexec bash %q\n' "$GROWMTP_ENTRYPOINT"
    } > "$resume_script"
    chmod 700 "$resume_script"
}
write_resume_config

TRAIN_LOG="$RUN_DIR/logs/training.log"
export GROWMTP_LOG_LEVEL="$LOG_LEVEL" GROWMTP_RUN_DIR="$RUN_DIR"
printf '===== Run invocation started %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$RUN_DIR/config/launch.txt"
printf '%s\n' \
    "run_dir=$RUN_DIR" \
    "ray_temp_dir=$RAY_TEMP_DIR" \
    "run_mode=$RUN_MODE" \
    "gpu_ids=${GPU_IDS:-CUDA_VISIBLE_DEVICES}" \
    "b200_min_memory_mib=$B200_MIN_MEMORY_MIB" \
    "train_gpus=$TRAIN_GPUS" \
    "run_action=$RUN_ACTION" \
    "base_model=$BASE_MODEL" \
    "prompt_key=$PROMPT_KEY" \
    "prepared_model=$PREPARED_MODEL_DIR" \
    "prepared_model_reused=$PREPARED_MODEL_REUSE" \
    "checkpoint_dir=$CHECKPOINT_DIR" \
    "train_file=$TRAIN_FILE" \
    "validation_file=$VAL_FILE" \
    "steps=$TRAIN_STEPS" \
    "response_length=$RESPONSE_LENGTH" \
    "max_prompt_length=$MAX_PROMPT_LENGTH" \
    "train_batch_size=$TRAIN_BATCH_SIZE" \
    "rollout_n=$ROLLOUT_N" \
    "dataloader_workers=$DATALOADER_NUM_WORKERS" \
    "save_freq=$SAVE_FREQ" \
    "test_freq=$TEST_FREQ" \
    "save_generations=$SAVE_GENERATIONS" \
    "mtp_probe_freq=$MTP_PROBE_FREQ" \
    "mtp_probe_max_cycles=$MTP_PROBE_MAX_CYCLES" \
    "mtp_probe_max_context=$MTP_PROBE_MAX_CONTEXT" \
    "mtp_refresh_fraction=$MTP_REFRESH_FRACTION" \
    "mtp_aux_ce_lambda=$MTP_AUX_CE_LAMBDA" \
    "mtp_aux_advantage_clip=$MTP_AUX_ADVANTAGE_CLIP" \
    "comparison_log_trajectories=$COMPARISON_LOG_TRAJECTORIES" \
    "final_validation=$FINAL_VALIDATION" \
    "val_before_train=$VAL_BEFORE_TRAIN" \
    "val_samples=$VAL_SAMPLES" \
    "lora_rank=$LORA_RANK" \
    "lora_alpha=$LORA_ALPHA" \
    "target_modules=$TARGET_MODULES_JSON" \
    "ar_baseline_ms_per_token=${AR_BASELINE_MS_PER_TOKEN:-unset}" \
    "ar_baseline_tokens_per_second_per_gpu=${AR_BASELINE_TOKENS_PER_SECOND_PER_GPU:-unset}" \
    "ar_baseline_auto=$AR_BASELINE_AUTO" \
    "ar_baseline_requests=$AR_BASELINE_REQUESTS" \
    "ar_baseline_repeats=$AR_BASELINE_REPEATS" \
    "ar_baseline_tp_size_per_replica=$AR_BASELINE_TP_SIZE_PER_REPLICA" \
    "rollout_gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    "attention=$ATTN_IMPLEMENTATION" \
    "log_level=$LOG_LEVEL" \
    "started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$RUN_DIR/config/launch.txt"

printf '\n╭─ GrowMTP Qwen3-4B LoRA ──────────────────────────────\n'
printf '│ Action      %s\n' "$RUN_ACTION"
printf '│ Mode        %s (%s steps)\n' "$RUN_MODE" "$TRAIN_STEPS"
printf '│ GPU IDs     %s\n' "${GPU_IDS:-CUDA_VISIBLE_DEVICES}"
printf '│ Train GPUs  %s\n' "$TRAIN_GPUS"
printf '│ LoRA        rank %s · alpha %s · targets %s\n' "$LORA_RANK" "$LORA_ALPHA" "$TARGET_MODULES_JSON"
printf '│ Comparison  probe freq %s · cycles %s · context %s · records %s\n' \
    "$MTP_PROBE_FREQ" "$MTP_PROBE_MAX_CYCLES" "$MTP_PROBE_MAX_CONTEXT" "$COMPARISON_LOG_TRAJECTORIES"
printf '│ Aux CE      lambda %s · advantage clip %s\n' "$MTP_AUX_CE_LAMBDA" "$MTP_AUX_ADVANTAGE_CLIP"
printf '│ Validation  before %s · final %s · samples %s · periodic %s\n' \
    "$VAL_BEFORE_TRAIN" "$FINAL_VALIDATION" "$VAL_SAMPLES" "$TEST_FREQ"
printf '│ Run folder  %s\n' "$RUN_DIR"
printf '│ Console     %s\n' "$LOG_LEVEL"
printf '│ Full log    %s\n' "$TRAIN_LOG"
printf '│ Metrics     %s\n' "$RUN_DIR/logs/metrics.jsonl"
printf '╰──────────────────────────────────────────────────────\n'

if (( STOP_REQUESTED )); then
    printf 'finished_utc=%s\nresult=cancelled_before_training\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$RUN_DIR/config/launch.txt"
    exit 0
fi

if [[ "$RUN_ACTION" == "fresh" && "$PREPARED_MODEL_REUSE" == 0 ]]; then
    printf '\nPreparing a fresh Qwen3-4B GrowMTP checkpoint at:\n%s\n' "$PREPARED_MODEL_DIR"
    GROWMTP_PYTHON="$GROWMTP_PYTHON" bash "$REPO_ROOT/scripts/prepare_model.sh" \
        --model-path "$BASE_MODEL" \
        --output "$PREPARED_MODEL_DIR"
elif [[ "$RUN_ACTION" == "fresh" ]]; then
    printf '\nReusing prepared model and restarting training at step 0:\n%s\n' "$PREPARED_MODEL_DIR"
else
    printf '\nResuming GrowMTP checkpoint at step %s from:\n%s\n' "$CHECKPOINT_STEP" "$CHECKPOINT_DIR"
fi

AR_BASELINE_JSON="$RUN_DIR/artifacts/ar_baseline.json"
AR_BASELINE_SOURCE="unset"
AR_BASELINE_RESOLVED_MS_PER_TOKEN=""
if [[ -n "$AR_BASELINE_MS_PER_TOKEN" ]]; then
    AR_BASELINE_SOURCE="environment_ms_per_token"
    AR_BASELINE_RESOLVED_MS_PER_TOKEN="$AR_BASELINE_MS_PER_TOKEN"
elif [[ -n "$AR_BASELINE_TOKENS_PER_SECOND_PER_GPU" ]]; then
    AR_BASELINE_SOURCE="environment_tokens_per_second_per_gpu"
    AR_BASELINE_RESOLVED_MS_PER_TOKEN="$("$GROWMTP_PYTHON" -c \
        'import sys; print(1000.0 / (float(sys.argv[1]) * int(sys.argv[2])))' \
        "$AR_BASELINE_TOKENS_PER_SECOND_PER_GPU" "$TRAIN_GPUS")"
elif [[ "$AR_BASELINE_AUTO" == "1" ]]; then
    if [[ -s "$AR_BASELINE_JSON" ]]; then
        AR_BASELINE_SOURCE="saved_ar_benchmark"
        AR_BASELINE_MS_PER_TOKEN="$("$GROWMTP_PYTHON" -c \
            'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["ms_per_token"])' \
            "$AR_BASELINE_JSON")"
    else
        AR_BASELINE_SOURCE="automatic_sglang_ar_benchmark"
        printf '\nRunning autoregressive SGLang baseline before GrowMTP training.\n'
        AR_BASELINE_LOG="$RUN_DIR/logs/ar_baseline.log"
        BASELINE_PYTHONPATH="$REPO_ROOT/verl:$REPO_ROOT/sglang/python"
        if [[ -v PYTHONPATH && -n "$PYTHONPATH" ]]; then
            BASELINE_PYTHONPATH="$BASELINE_PYTHONPATH:$PYTHONPATH"
        fi
        PYTHONPATH="$BASELINE_PYTHONPATH" "$GROWMTP_PYTHON" "$REPO_ROOT/scripts/benchmark_ar_baseline.py" \
            --model-path "$PREPARED_MODEL_DIR" \
            --train-file "$TRAIN_FILE" \
            --prompt-key "$PROMPT_KEY" \
            --num-prompts "$AR_BASELINE_REQUESTS" \
            --response-length "$RESPONSE_LENGTH" \
            --max-prompt-length "$MAX_PROMPT_LENGTH" \
            --repeats "$AR_BASELINE_REPEATS" \
            --gpus "$TRAIN_GPUS" \
            --tp-per-replica "$AR_BASELINE_TP_SIZE_PER_REPLICA" \
            --memory-fraction "$ROLLOUT_GPU_MEMORY_UTILIZATION" \
            --output "$AR_BASELINE_JSON" 2>&1 | tee -a "$AR_BASELINE_LOG"
        AR_BASELINE_MS_PER_TOKEN="$("$GROWMTP_PYTHON" -c \
            'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["ms_per_token"])' \
            "$AR_BASELINE_JSON")"
    fi
    "$GROWMTP_PYTHON" -c 'import math, sys; value=float(sys.argv[1]); sys.exit(0 if math.isfinite(value) and value > 0 else 1)' \
        "$AR_BASELINE_MS_PER_TOKEN" || die "AR baseline artifact has an invalid ms_per_token value"
    AR_BASELINE_RESOLVED_MS_PER_TOKEN="$AR_BASELINE_MS_PER_TOKEN"
    EXTRA_HYDRA_OVERRIDES+=(
        "actor_rollout_ref.model.mtp.ar_baseline_ms_per_token=$AR_BASELINE_MS_PER_TOKEN"
    )
fi

if [[ -n "$AR_BASELINE_RESOLVED_MS_PER_TOKEN" ]]; then
    printf 'ar_baseline_source=%s\nar_baseline_ms_per_token=%s\n' \
        "$AR_BASELINE_SOURCE" "$AR_BASELINE_RESOLVED_MS_PER_TOKEN" >> "$RUN_DIR/config/launch.txt"
    "$GROWMTP_PYTHON" - "$RUN_DIR/logs/metrics.jsonl" "$AR_BASELINE_RESOLVED_MS_PER_TOKEN" \
        "$TRAIN_GPUS" <<'PY'
import datetime
import json
import math
import pathlib
import sys

metrics_path = pathlib.Path(sys.argv[1])
ms_per_token = float(sys.argv[2])
gpu_count = int(sys.argv[3])
row = {
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "step": 0,
    "phase": "ar_baseline",
    "target/ms_per_generated_token": ms_per_token,
    "perf/rollout_tokens_per_second_per_gpu": 1000.0 / (ms_per_token * gpu_count),
}
if not math.isfinite(ms_per_token) or ms_per_token <= 0:
    raise SystemExit("Resolved AR baseline ms/token must be positive and finite")
metrics_path.parent.mkdir(parents=True, exist_ok=True)
existing = []
if metrics_path.exists():
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        try:
            existing.append(json.loads(line))
        except json.JSONDecodeError:
            continue
if not any(
    item.get("phase") == "ar_baseline"
    and item.get("target/ms_per_generated_token") == ms_per_token
    for item in existing
):
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
PY
fi

if (( STOP_REQUESTED )); then
    printf 'finished_utc=%s\nresult=cancelled_before_training\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$RUN_DIR/config/launch.txt"
    exit 0
fi

mkdir -p "$CHECKPOINT_DIR"
rm -f "$CHECKPOINT_DIR/.stop_after_step"
TRAIN_COMMAND=(
    bash "$REPO_ROOT/scripts/train.sh"
    --model qwen3_4b
    --model-path "$PREPARED_MODEL_DIR"
    --train-file "$TRAIN_FILE"
    --val-file "$VAL_FILE"
    --output "$CHECKPOINT_DIR"
    --gpus "$TRAIN_GPUS"
    --nodes 1
    --steps "$TRAIN_STEPS"
    --response-length "$RESPONSE_LENGTH"
    --prompt-key "$PROMPT_KEY"
    --depth 5
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "actor_rollout_ref.model.lora_rank=$LORA_RANK"
    "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA"
    actor_rollout_ref.model.lora.merge=true
    "actor_rollout_ref.model.target_modules=$TARGET_MODULES_JSON"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.dataloader_num_workers=$DATALOADER_NUM_WORKERS"
    "actor_rollout_ref.rollout.n=$ROLLOUT_N"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$AR_BASELINE_TP_SIZE_PER_REPLICA"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION"
    "actor_rollout_ref.rollout.agent.num_workers=$AGENT_LOOP_WORKERS"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_BATCHED_TOKENS"
    "actor_rollout_ref.rollout.val_kwargs.n=$VAL_SAMPLES"
    "++actor_rollout_ref.model.mtp.comparison_probe_frequency=$MTP_PROBE_FREQ"
    "++actor_rollout_ref.model.mtp.comparison_probe_max_cycles=$MTP_PROBE_MAX_CYCLES"
    "++actor_rollout_ref.model.mtp.comparison_probe_max_context=$MTP_PROBE_MAX_CONTEXT"
    "++actor_rollout_ref.model.mtp.comparison_refresh_fraction=$MTP_REFRESH_FRACTION"
    "++actor_rollout_ref.model.mtp.comparison_log_trajectories=$COMPARISON_LOG_TRAJECTORIES_HYDRA"
    "++actor_rollout_ref.model.mtp.rollout_aux_ce_lambda=$MTP_AUX_CE_LAMBDA"
    "++actor_rollout_ref.model.mtp.rollout_aux_advantage_clip=$MTP_AUX_ADVANTAGE_CLIP"
    "++trainer.final_validation=$FINAL_VALIDATION_HYDRA"
    "trainer.val_before_train=$VAL_BEFORE_TRAIN_HYDRA"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.resume_mode=$RESUME_MODE"
)
TRAIN_COMMAND+=(
    "++ray_kwargs.ray_init._temp_dir=$RAY_TEMP_DIR"
    "global_profiler.save_path=$RUN_DIR/artifacts/profiling"
)
if [[ "$SAVE_GENERATIONS" == 1 ]]; then
    TRAIN_COMMAND+=(
        "trainer.rollout_data_dir=$RUN_DIR/artifacts/rollouts"
        "trainer.validation_data_dir=$RUN_DIR/artifacts/validation"
    )
fi
TRAIN_COMMAND+=("${EXTRA_HYDRA_OVERRIDES[@]}")

setsid env GROWMTP_PYTHON="$GROWMTP_PYTHON" GROWMTP_LOG_LEVEL="$LOG_LEVEL" \
    "$GROWMTP_PYTHON" "$REPO_ROOT/scripts/run_logged.py" \
    --log-file "$TRAIN_LOG" --level "$LOG_LEVEL" -- "${TRAIN_COMMAND[@]}" &
TRAIN_PID=$!
if (( STOP_REQUESTED )); then
    : > "$CHECKPOINT_DIR/.stop_after_step"
fi

set +e
while :; do
    wait "$TRAIN_PID"
    TRAIN_STATUS=$?
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        break
    fi
done
set -e
TRAIN_PID=""
trap - INT TERM
rm -f "$CHECKPOINT_DIR/.stop_after_step"

if (( TRAIN_STATUS != 0 )); then
    RAY_SESSION_TARGET="$(readlink "$RAY_TEMP_DIR/session_latest" 2>/dev/null || true)"
    RAY_SESSION_NAME="${RAY_SESSION_TARGET##*/}"
    [[ -n "$RAY_SESSION_NAME" ]] || RAY_SESSION_NAME=unknown
    RAY_SESSION_LOGS="$RAY_TEMP_DIR/session_latest/logs"
    if [[ -d "$RAY_SESSION_LOGS" ]]; then
        RAY_LOG_DIR="$RUN_DIR/logs/ray-$RAY_SESSION_NAME"
        mkdir -p "$RAY_LOG_DIR"
        if cp -a "$RAY_SESSION_LOGS/." "$RAY_LOG_DIR/"; then
            printf 'Ray session logs copied to: %s\n' "$RAY_LOG_DIR"
        else
            printf 'WARNING: could not copy Ray logs from %s\\n' "$RAY_SESSION_LOGS" >&2
        fi
    else
        printf 'Ray session logs were not found under %s\\n' "$RAY_SESSION_LOGS" >&2
    fi
    printf 'finished_utc=%s\nresult=failed\nexit_code=%s\n' \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$TRAIN_STATUS" >> "$RUN_DIR/config/launch.txt"
    printf 'Training process exited with status %s.\n' "$TRAIN_STATUS" >&2
    exit "$TRAIN_STATUS"
fi

if [[ -f "$CHECKPOINT_DIR/.suspend_requested" ]]; then
    RUN_RESULT="suspended"
    printf '\nRun suspended safely at checkpoint step %s. Resume this run with:\n' \
        "$(< "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt")"
    printf 'Resume this run with: bash %q\n' "$RUN_DIR/config/resume.sh"
else
    RUN_RESULT="complete"
    printf '\nRun complete. Run folder: %s\n' "$RUN_DIR"
fi
printf 'finished_utc=%s\nresult=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$RUN_RESULT" >> "$RUN_DIR/config/launch.txt"
printf 'Full training log: %s\nMetrics: %s\nLauncher log: %s\n' \
    "$TRAIN_LOG" "$RUN_DIR/logs/metrics.jsonl" "$RUN_DIR/logs/launcher.log"
