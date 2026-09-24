#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${MODAL_SMOKE_LOG:-${REPO_ROOT}/logs/modal-growmtp-smoke-${STAMP}.log}"

printf 'Modal run log: %s\n' "$LOG_FILE"
set +e
modal run --timestamps "${SCRIPT_DIR}/modal_growmtp_smoke_v2.py" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e
exit "$STATUS"
