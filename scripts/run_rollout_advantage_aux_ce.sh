#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -z "${MTP_AUX_CE_LAMBDA:-}" ]]; then
  printf 'ERROR: Set MTP_AUX_CE_LAMBDA from the pilot gradient-scale calibration.\n' >&2
  exit 2
fi

export MTP_AUX_CE_REQUIRED=1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_policy_shift_matching_baseline.sh" "$@"
