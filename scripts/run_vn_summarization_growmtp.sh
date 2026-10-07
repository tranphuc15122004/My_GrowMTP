#!/usr/bin/env bash
set -euo pipefail

# Run GrowMTP on Parquet prepared from id/text/summary Vietnamese JSONL.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${DATA_DIR:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/datasets/vn_summarization/v1}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/validation.parquet}"
RUN_BASE_DIR="${RUN_BASE_DIR:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/vn-summarization-growmtp/runs}"
RUN_MODE="${RUN_MODE:-smoke}"
GROWMTP_SEED="${GROWMTP_SEED:-1}"
GROWMTP_DATA_MANIFEST="${GROWMTP_DATA_MANIFEST:-$DATA_DIR/manifest.json}"

GROWMTP_REWARD_FILE="$REPO_ROOT/scripts/vn_summarization_reward.py"
GROWMTP_ENABLE_THINKING=0
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
case "$RUN_MODE" in
    smoke)
        RESPONSE_LENGTH="${RESPONSE_LENGTH:-256}"
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
        ROLLOUT_N="${ROLLOUT_N:-2}"
        ;;
    pilot|full)
        RESPONSE_LENGTH="${RESPONSE_LENGTH:-512}"
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
        ROLLOUT_N="${ROLLOUT_N:-4}"
        ;;
    *) printf 'ERROR: RUN_MODE must be smoke, pilot, or full\n' >&2; exit 2 ;;
esac
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
[[ "$MAX_PROMPT_LENGTH" =~ ^[1-9][0-9]*$ ]] || { printf 'ERROR: MAX_PROMPT_LENGTH must be a positive integer\n' >&2; exit 2; }
[[ "$RESPONSE_LENGTH" =~ ^[1-9][0-9]*$ ]] || { printf 'ERROR: RESPONSE_LENGTH must be a positive integer\n' >&2; exit 2; }
REQUIRED_TOKEN_LEN_PER_GPU=$((MAX_PROMPT_LENGTH + RESPONSE_LENGTH))
if [[ -z "${MAX_TOKEN_LEN_PER_GPU:-}" ]]; then
    MAX_TOKEN_LEN_PER_GPU=8192
    (( REQUIRED_TOKEN_LEN_PER_GPU <= MAX_TOKEN_LEN_PER_GPU )) || MAX_TOKEN_LEN_PER_GPU="$REQUIRED_TOKEN_LEN_PER_GPU"
else
    (( MAX_TOKEN_LEN_PER_GPU >= REQUIRED_TOKEN_LEN_PER_GPU )) || {
        printf 'ERROR: MAX_TOKEN_LEN_PER_GPU must be at least MAX_PROMPT_LENGTH + RESPONSE_LENGTH (%s)\n' \
            "$REQUIRED_TOKEN_LEN_PER_GPU" >&2
        exit 2
    }
fi
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
VAL_SAMPLES="${VAL_SAMPLES:-4}"

# Plain GrowMTP by default; a positive MTP_AUX_CE_LAMBDA enables the Idea arm.
MTP_PROBE_FREQ=0
MTP_REFRESH_FRACTION=0
MTP_AUX_CE_LAMBDA="${MTP_AUX_CE_LAMBDA:-0}"
if awk -v value="$MTP_AUX_CE_LAMBDA" 'BEGIN { exit !(value + 0 > 0) }'; then
    RUN_LABEL="${RUN_LABEL:-idea-rollout-adv-ce}"
else
    RUN_LABEL="${RUN_LABEL:-growmtp-baseline}"
fi
if [[ "$RUN_MODE" == full ]]; then
    FINAL_VALIDATION="${FINAL_VALIDATION:-1}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-1}"
else
    FINAL_VALIDATION="${FINAL_VALIDATION:-0}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-0}"
fi

export DATA_DIR TRAIN_FILE VAL_FILE RUN_BASE_DIR RUN_MODE GROWMTP_REWARD_FILE
export GROWMTP_SEED GROWMTP_DATA_MANIFEST RUN_LABEL
export GROWMTP_ENABLE_THINKING MAX_PROMPT_LENGTH RESPONSE_LENGTH TRAIN_BATCH_SIZE
export PPO_MINI_BATCH_SIZE ROLLOUT_N MAX_BATCHED_TOKENS MAX_TOKEN_LEN_PER_GPU
export VAL_MAX_SAMPLES VAL_SAMPLES MTP_PROBE_FREQ MTP_REFRESH_FRACTION
export MTP_AUX_CE_LAMBDA FINAL_VALIDATION VAL_BEFORE_TRAIN

exec bash "$SCRIPT_DIR/run_b200_growmtp_lora_server.sh" "$@"
