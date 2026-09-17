#!/usr/bin/env bash
# Uses the pinned CUDA environment as a base; installs both local Python components.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: install.sh [--check]; see README for CUDA dependencies.'
    exit 0
fi
if [[ "${1:-}" != --check ]]; then
    "$GROWMTP_PY" -m venv --system-site-packages "$GROWMTP_ROOT/.venv"
    "$GROWMTP_ROOT/.venv/bin/python" -m pip install --no-deps --no-build-isolation \
        -e "$GROWMTP_ROOT/sglang/python" -e "$GROWMTP_ROOT/verl"
    GROWMTP_PY="$GROWMTP_ROOT/.venv/bin/python"
fi
"$GROWMTP_PY" -m verl.trainer.mtp.launch check
