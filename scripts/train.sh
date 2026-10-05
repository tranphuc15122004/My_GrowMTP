#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
source "$SCRIPT_DIR/gpu_args.sh"

TRAIN_ARGS=()
REQUESTED_GPUS=""
CLI_GPU_IDS=""
HAS_CLI_GPU_IDS=0
while (($#)); do
    case "$1" in
        --gpuid)
            (( $# >= 2 )) || { printf 'ERROR: --gpuid requires a comma-separated list, for example 2,3\n' >&2; exit 2; }
            (( HAS_CLI_GPU_IDS == 0 )) || { printf 'ERROR: specify --gpuid only once\n' >&2; exit 2; }
            CLI_GPU_IDS="$2"
            HAS_CLI_GPU_IDS=1
            shift 2
            ;;
        --gpuid=*)
            (( HAS_CLI_GPU_IDS == 0 )) || { printf 'ERROR: specify --gpuid only once\n' >&2; exit 2; }
            CLI_GPU_IDS="${1#*=}"
            HAS_CLI_GPU_IDS=1
            shift
            ;;
        --gpus)
            (( $# >= 2 )) || { printf 'ERROR: --gpus requires a count\n' >&2; exit 2; }
            REQUESTED_GPUS="$2"
            TRAIN_ARGS+=("$1" "$2")
            shift 2
            ;;
        --gpus=*)
            REQUESTED_GPUS="${1#*=}"
            TRAIN_ARGS+=("$1")
            shift
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

if (( HAS_CLI_GPU_IDS )); then
    parse_gpuid_args --gpuid "$CLI_GPU_IDS" || exit $?
else
    parse_gpuid_args || exit $?
fi
if [[ -n "$GPU_IDS" ]]; then
    if [[ -n "$REQUESTED_GPUS" && "$REQUESTED_GPUS" != "$GPU_COUNT" ]]; then
        printf 'ERROR: --gpus=%s does not match %s selected GPU(s) in --gpuid %s\n' \
            "$REQUESTED_GPUS" "$GPU_COUNT" "$GPU_IDS" >&2
        exit 2
    fi
    command -v nvidia-smi >/dev/null 2>&1 || { printf 'ERROR: nvidia-smi is required to select physical GPU IDs\n' >&2; exit 2; }
    GPU_SUMMARY="$(nvidia-smi --id="$GPU_IDS" --query-gpu=index --format=csv,noheader 2>/dev/null)" || {
        printf 'ERROR: requested GPU(s) are not available: %s\n' "$GPU_IDS" >&2
        exit 2
    }
    GPU_SUMMARY_COUNT="$(printf '%s\n' "$GPU_SUMMARY" | awk 'NF { count++ } END { print count+0 }')"
    [[ "$GPU_SUMMARY_COUNT" == "$GPU_COUNT" ]] || {
        printf 'ERROR: nvidia-smi found %s GPU(s) for --gpuid %s\n' "$GPU_SUMMARY_COUNT" "$GPU_IDS" >&2
        exit 2
    }
    if [[ -z "$REQUESTED_GPUS" ]]; then
        TRAIN_ARGS+=(--gpus "$GPU_COUNT")
    fi
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi

exec "$GROWMTP_PY" -m verl.trainer.mtp.launch train "${TRAIN_ARGS[@]}"
