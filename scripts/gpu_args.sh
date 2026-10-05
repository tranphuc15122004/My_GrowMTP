#!/usr/bin/env bash

# Shared --gpuid parsing for local training launchers. GPU indices are physical
# host indices; CUDA_VISIBLE_DEVICES remaps them inside the training process.
parse_gpuid_args() {
    local cli_gpu_ids=""
    local has_cli_gpu_ids=0
    local gpu_id

    while (($#)); do
        case "$1" in
            --gpuid)
                (( $# >= 2 )) || {
                    printf 'ERROR: --gpuid requires a comma-separated GPU list, for example --gpuid 2,3\n' >&2
                    return 2
                }
                (( has_cli_gpu_ids == 0 )) || {
                    printf 'ERROR: specify --gpuid only once\n' >&2
                    return 2
                }
                cli_gpu_ids="$2"
                has_cli_gpu_ids=1
                shift 2
                ;;
            --gpuid=*)
                (( has_cli_gpu_ids == 0 )) || {
                    printf 'ERROR: specify --gpuid only once\n' >&2
                    return 2
                }
                cli_gpu_ids="${1#*=}"
                has_cli_gpu_ids=1
                shift
                ;;
            *)
                printf 'ERROR: unsupported launcher argument: %s (supported option: --gpuid 2,3)\n' "$1" >&2
                return 2
                ;;
        esac
    done

    if (( has_cli_gpu_ids )); then
        GPU_IDS="$cli_gpu_ids"
    fi
    GPU_IDS="${GPU_IDS:-}"
    if [[ -z "$GPU_IDS" ]]; then
        GPU_COUNT=0
        return 0
    fi
    [[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
        printf 'ERROR: GPU IDs must be comma-separated physical indices, for example 2,3\n' >&2
        return 2
    }

    GPU_ID_ARRAY=()
    local -A seen_gpu_ids=()
    IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
    for gpu_id in "${GPU_ID_ARRAY[@]}"; do
        [[ -z "${seen_gpu_ids[$gpu_id]:-}" ]] || {
            printf 'ERROR: GPU list contains duplicate index %s\n' "$gpu_id" >&2
            return 2
        }
        seen_gpu_ids[$gpu_id]=1
    done
    GPU_COUNT="${#GPU_ID_ARRAY[@]}"
}
