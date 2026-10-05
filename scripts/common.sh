#!/usr/bin/env bash
set -euo pipefail
GROWMTP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$GROWMTP_ROOT/scripts/python_env.sh"
resolve_growmtp_python "$GROWMTP_ROOT" || { printf 'ERROR: no Python interpreter found in the active environment or PATH\n' >&2; exit 2; }
GROWMTP_PY="$GROWMTP_PYTHON"
export PYTHONPATH="$GROWMTP_ROOT/verl:$GROWMTP_ROOT/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
