#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec "$GROWMTP_PY" -m verl.trainer.mtp.launch train "$@"
