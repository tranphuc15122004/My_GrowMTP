#!/usr/bin/env bash
set -Eeuo pipefail

# Defaults from the user's B200 server. Override any variable in the shell
# before invoking this wrapper if a path or output destination changes.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GROWMTP_PYTHON="${GROWMTP_PYTHON:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/phucvenv/bin/python}"
DATA_DIR="${DATA_DIR:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/DAPO-math-17.4K}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/validation.parquet}"
BASE_MODEL="${BASE_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B}"
PREPARED_MODEL_DIR="${PREPARED_MODEL_DIR:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/models/Qwen3-4B-growmtp-b200-smoke}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3-4b-growmtp-lora-b200-smoke}"

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

export GROWMTP_PYTHON DATA_DIR TRAIN_FILE VAL_FILE BASE_MODEL PREPARED_MODEL_DIR RUN_OUTPUT_DIR
export PYTHONPATH="$REPO_ROOT/verl:$REPO_ROOT/sglang/python${PYTHONPATH:+:$PYTHONPATH}"

printf 'Preflighting the real trainer before any model preparation/download.\n'
"$GROWMTP_PYTHON" "$SCRIPT_DIR/check_training_imports.py"
exec bash "$SCRIPT_DIR/run_b200_growmtp_lora.sh"
