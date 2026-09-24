#!/usr/bin/env bash
set -Eeuo pipefail

# Edit the values in this section for your B200 server. Every output path
# must be new: this script intentionally refuses to reuse checkpoints.
GROWMTP_PYTHON="${GROWMTP_PYTHON:-/home/tuantb/fast_infer_text_sum/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/DAPO-math-17.4K}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/test.parquet}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
PREPARED_MODEL_DIR="${PREPARED_MODEL_DIR:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/models/Qwen3-4B-growmtp-fresh}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3-4b-growmtp-lora-fresh}"

# smoke: 2 short steps. full: repo preset (500 steps, 8192 response tokens).
RUN_MODE="${RUN_MODE:-smoke}"
# Keep PEFT training, but merge the adapter into rollout weights because
# SGLang's dynamic-LoRA path rejects GrowMTP's EAGLE speculative decoding.
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
TARGET_MODULES_JSON="${TARGET_MODULES_JSON:-[\"q_proj\",\"v_proj\"]}"

# For full runs, keep this at -1 when VAL_FILE is your held-out test set.
# Set it to e.g. 50 only when VAL_FILE is a separate validation split.
FULL_TEST_FREQ="${FULL_TEST_FREQ:--1}"
REQUIRE_B200="${REQUIRE_B200:-1}"

# Add repo-specific Hydra overrides here if needed, for example:
# EXTRA_HYDRA_OVERRIDES+=(actor_rollout_ref.actor.optim.lr=1e-6)
EXTRA_HYDRA_OVERRIDES=()

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

[[ -x "$GROWMTP_PYTHON" ]] || die "Python executable not found: $GROWMTP_PYTHON"
[[ -f "$TRAIN_FILE" ]] || die "Training parquet not found: $TRAIN_FILE"
[[ -f "$VAL_FILE" ]] || die "Validation/test parquet not found: $VAL_FILE"
[[ "$LORA_RANK" =~ ^[1-9][0-9]*$ ]] || die "LORA_RANK must be a positive integer"
[[ "$LORA_ALPHA" =~ ^[1-9][0-9]*$ ]] || die "LORA_ALPHA must be a positive integer"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not available; run this script on the GPU server"
GPU_SUMMARY="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
printf 'Visible GPUs:\n%s\n' "$GPU_SUMMARY"

export REQUIRE_B200
"$GROWMTP_PYTHON" - <<'PY'
import os
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the selected Python environment")
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print("PyTorch CUDA devices:", names)
if not torch.cuda.is_bf16_supported():
    raise SystemExit("This GrowMTP setup expects a GPU with BF16 support")
if os.environ.get("REQUIRE_B200", "1") == "1" and not any("B200" in name.upper() for name in names):
    raise SystemExit("No B200 is visible to this Python process; refusing to launch")
PY

printf '\nChecking GrowMTP dependencies with %s\n' "$GROWMTP_PYTHON"
GROWMTP_PYTHON="$GROWMTP_PYTHON" bash "$REPO_ROOT/scripts/install.sh" --check

"$GROWMTP_PYTHON" - "$TRAIN_FILE" "$VAL_FILE" <<'PY'
import sys
import pyarrow.parquet as pq

required = {"prompt", "reward_model", "data_source"}
for path in sys.argv[1:]:
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
        0, columns=["prompt", "reward_model", "data_source"]
    ).slice(0, 1).to_pylist()[0]
    prompt = sample["prompt"]
    reward = sample["reward_model"]
    if not isinstance(prompt, list) or not prompt or not prompt[0].get("content"):
        raise SystemExit(f"{path}: prompt must be a non-empty chat-message list")
    if not isinstance(reward, dict) or not reward.get("ground_truth"):
        raise SystemExit(f"{path}: reward_model.ground_truth is missing")
    print(f"Dataset OK: {path} ({parquet.metadata.num_rows:,} rows)")
PY

[[ ! -e "$PREPARED_MODEL_DIR" ]] || die "Prepared-model path already exists; choose a new path for a fresh MTP head: $PREPARED_MODEL_DIR"
[[ ! -e "$RUN_OUTPUT_DIR" ]] || die "Run-output path already exists; choose a new path to guarantee a fresh run: $RUN_OUTPUT_DIR"

case "$RUN_MODE" in
    smoke)
        TRAIN_STEPS=2
        RESPONSE_LENGTH=512
        TRAIN_BATCH_SIZE=2
        ROLLOUT_N=2
        PPO_MINI_BATCH_SIZE=2
        MAX_BATCHED_TOKENS=4096
        MAX_TOKEN_LEN_PER_GPU=8192
        SAVE_FREQ=2
        TEST_FREQ=-1
        ;;
    full)
        TRAIN_STEPS=500
        RESPONSE_LENGTH=8192
        TRAIN_BATCH_SIZE=64
        ROLLOUT_N=8
        PPO_MINI_BATCH_SIZE=64
        MAX_BATCHED_TOKENS=32768
        MAX_TOKEN_LEN_PER_GPU=32768
        SAVE_FREQ=50
        TEST_FREQ="$FULL_TEST_FREQ"
        ;;
    *)
        die "RUN_MODE must be 'smoke' or 'full'"
        ;;
esac

printf '\nPreparing a fresh Qwen3-4B GrowMTP checkpoint at:\n%s\n' "$PREPARED_MODEL_DIR"
GROWMTP_PYTHON="$GROWMTP_PYTHON" bash "$REPO_ROOT/scripts/prepare_model.sh" \
    --model-path "$BASE_MODEL" \
    --output "$PREPARED_MODEL_DIR"

printf '\nStarting fresh LoRA run: mode=%s steps=%s rank=%s alpha=%s\n' \
    "$RUN_MODE" "$TRAIN_STEPS" "$LORA_RANK" "$LORA_ALPHA"
GROWMTP_PYTHON="$GROWMTP_PYTHON" bash "$REPO_ROOT/scripts/train.sh" \
    --model qwen3_4b \
    --model-path "$PREPARED_MODEL_DIR" \
    --train-file "$TRAIN_FILE" \
    --val-file "$VAL_FILE" \
    --output "$RUN_OUTPUT_DIR" \
    --gpus 1 \
    --nodes 1 \
    --steps "$TRAIN_STEPS" \
    --response-length "$RESPONSE_LENGTH" \
    --depth 5 \
    "actor_rollout_ref.model.lora_rank=$LORA_RANK" \
    "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA" \
    actor_rollout_ref.model.lora.merge=true \
    "actor_rollout_ref.model.target_modules=$TARGET_MODULES_JSON" \
    "data.train_batch_size=$TRAIN_BATCH_SIZE" \
    "actor_rollout_ref.rollout.n=$ROLLOUT_N" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" \
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" \
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    "trainer.save_freq=$SAVE_FREQ" \
    "trainer.test_freq=$TEST_FREQ" \
    trainer.resume_mode=disable \
    "${EXTRA_HYDRA_OVERRIDES[@]}"

printf '\nRun complete. Output: %s\n' "$RUN_OUTPUT_DIR"
