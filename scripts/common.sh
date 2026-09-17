#!/usr/bin/env bash
set -euo pipefail
GROWMTP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${GROWMTP_PYTHON:-}" ]]; then
    GROWMTP_PY="$GROWMTP_PYTHON"
elif [[ -x "$GROWMTP_ROOT/.venv/bin/python" ]]; then
    GROWMTP_PY="$GROWMTP_ROOT/.venv/bin/python"
else
    GROWMTP_PY=python3
fi
export PYTHONPATH="$GROWMTP_ROOT/verl:$GROWMTP_ROOT/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
