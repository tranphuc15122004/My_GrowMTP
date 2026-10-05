#!/usr/bin/env bash
set -Eeuo pipefail

# Defaults from the user's B200 server. Override any variable in the shell
# before invoking this wrapper if a path or output destination changes.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GROWMTP_PYTHON="${GROWMTP_PYTHON:-/home/tuantb/fast_infer_text_sum/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/DAPO-math-17.4K}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/validation.parquet}"
BASE_MODEL="${BASE_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B}"
RUN_BASE_DIR="${RUN_BASE_DIR:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3-4b-growmtp-lora/runs}"
RUN_DIR="${RUN_DIR:-}"
RUN_MODE="${RUN_MODE:-full}"
GPU_IDS="${GPU_IDS:-0,1}"
LOG_LEVEL="${LOG_LEVEL:-compact}"
SAVE_GENERATIONS="${SAVE_GENERATIONS:-0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
B200_MIN_MEMORY_MIB="${B200_MIN_MEMORY_MIB:-170000}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

source "$SCRIPT_DIR/gpu_args.sh"
parse_gpuid_args "$@" || exit $?

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

[[ "$B200_MIN_MEMORY_MIB" =~ ^[1-9][0-9]*$ ]] || \
    die "B200_MIN_MEMORY_MIB must be a positive integer"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not available; run this script on the B200 server"

for GPU_ID in "${GPU_ID_ARRAY[@]}"; do
    GPU_INFO="$(nvidia-smi --id="$GPU_ID" --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>/dev/null)" || \
        die "GPU index $GPU_ID is not available according to nvidia-smi"
    IFS=',' read -r ACTUAL_GPU_ID GPU_NAME GPU_MEMORY_MIB <<< "$GPU_INFO"
    ACTUAL_GPU_ID="$(trim "$ACTUAL_GPU_ID")"
    GPU_NAME="$(trim "$GPU_NAME")"
    GPU_MEMORY_MIB="$(trim "$GPU_MEMORY_MIB")"
    [[ "$ACTUAL_GPU_ID" == "$GPU_ID" ]] || \
        die "Requested GPU $GPU_ID resolved to unexpected device index '$ACTUAL_GPU_ID'"
    [[ "${GPU_NAME^^}" == *B200* ]] || \
        die "GPU $GPU_ID is '$GPU_NAME'; this launcher requires B200 cards"
    [[ "$GPU_MEMORY_MIB" =~ ^[0-9]+$ ]] || \
        die "Could not read total memory for GPU $GPU_ID: $GPU_INFO"
    (( GPU_MEMORY_MIB >= B200_MIN_MEMORY_MIB )) || \
        die "GPU $GPU_ID has ${GPU_MEMORY_MIB} MiB; expected a B200 with about 180 GB (minimum ${B200_MIN_MEMORY_MIB} MiB)"
    printf 'Selected GPU %s: %s, %s MiB\n' "$GPU_ID" "$GPU_NAME" "$GPU_MEMORY_MIB"
done

if [[ -n "${TRAIN_GPUS:-}" && "$TRAIN_GPUS" != "$GPU_COUNT" ]]; then
    die "TRAIN_GPUS=$TRAIN_GPUS does not match $GPU_COUNT GPU(s) in GPU_IDS=$GPU_IDS; configure GPU_IDS only"
fi
TRAIN_GPUS="$GPU_COUNT"
CUDA_VISIBLE_DEVICES="$GPU_IDS"
REQUIRE_B200="${REQUIRE_B200:-1}"
export GPU_IDS TRAIN_GPUS CUDA_VISIBLE_DEVICES REQUIRE_B200 RUN_MODE
export MAX_PROMPT_LENGTH ROLLOUT_GPU_MEMORY_UTILIZATION B200_MIN_MEMORY_MIB

[[ -x "$GROWMTP_PYTHON" ]] || {
    printf 'ERROR: Python executable not found: %s\n' "$GROWMTP_PYTHON" >&2
    exit 1
}
[[ -f "$TRAIN_FILE" ]] || {
    printf 'ERROR: Training parquet not found: %s\n' "$TRAIN_FILE" >&2
    exit 1
}
[[ -f "$VAL_FILE" ]] || {
    printf 'ERROR: Validation parquet not found: %s\n' "$VAL_FILE" >&2
    exit 1
}

GROWMTP_ENTRYPOINT="$SCRIPT_DIR/run_b200_growmtp_lora_server.sh"
export GROWMTP_PYTHON DATA_DIR TRAIN_FILE VAL_FILE BASE_MODEL RUN_BASE_DIR RUN_DIR LOG_LEVEL SAVE_GENERATIONS GROWMTP_ENTRYPOINT
export PYTHONPATH="$REPO_ROOT/verl:$REPO_ROOT/sglang/python${PYTHONPATH:+:$PYTHONPATH}"

SKIP_TRAINING_IMPORT_PREFLIGHT="${SKIP_TRAINING_IMPORT_PREFLIGHT:-0}"
case "$SKIP_TRAINING_IMPORT_PREFLIGHT" in
    0|1) ;;
    *) printf 'ERROR: SKIP_TRAINING_IMPORT_PREFLIGHT must be 0 or 1\n' >&2; exit 2 ;;
esac

run_preflight() {
    local status

    if [[ "$SKIP_TRAINING_IMPORT_PREFLIGHT" == "1" ]]; then
        printf 'Skipping trainer import check as requested; checking launch dependencies...\n'
    else
        printf 'Checking trainer imports...\n'
        "$GROWMTP_PYTHON" "$SCRIPT_DIR/check_training_imports.py"
        status=$?
        if (( status != 0 )); then
            return "$status"
        fi
        printf 'Checking GrowMTP and SGLang launch dependencies...\n'
    fi
    GROWMTP_PYTHON="$GROWMTP_PYTHON" bash "$SCRIPT_DIR/install.sh" --check
    status=$?
    if (( status != 0 )); then
        return "$status"
    fi
}

PREFLIGHT_STATUS=0
run_preflight || PREFLIGHT_STATUS=$?
if (( PREFLIGHT_STATUS == 130 || PREFLIGHT_STATUS == 143 )); then
    exit "$PREFLIGHT_STATUS"
fi
if (( PREFLIGHT_STATUS != 0 )); then
    printf 'GrowMTP dependencies are incomplete; installing the pinned runtime into %s.\n' "$GROWMTP_PYTHON"
    "$GROWMTP_PYTHON" "$SCRIPT_DIR/install_modal_image_deps.py" --repo-root "$REPO_ROOT"
    printf 'Rechecking the training environment after dependency installation.\n'
    run_preflight || exit "$?"
fi

printf 'Preflight passed; preparing the run.\n'
export GROWMTP_PREFLIGHT_PASSED=1
exec bash "$SCRIPT_DIR/run_b200_growmtp_lora.sh"
