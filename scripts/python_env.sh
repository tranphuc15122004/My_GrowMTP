#!/usr/bin/env bash

# Resolve one interpreter for launcher checks and training subprocesses. An
# explicit GROWMTP_PYTHON override wins; otherwise honor the active shell env.
resolve_growmtp_python() {
    local repo_root="$1"

    if [[ -n "${GROWMTP_PYTHON:-}" ]]; then
        return 0
    elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        GROWMTP_PYTHON="$VIRTUAL_ENV/bin/python"
    elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        GROWMTP_PYTHON="$CONDA_PREFIX/bin/python"
    elif command -v python >/dev/null 2>&1; then
        GROWMTP_PYTHON="$(command -v python)"
    elif [[ -x "$repo_root/.venv/bin/python" ]]; then
        GROWMTP_PYTHON="$repo_root/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        GROWMTP_PYTHON="$(command -v python3)"
    else
        return 1
    fi
}
