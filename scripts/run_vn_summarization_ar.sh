#!/usr/bin/env bash
# Run on the B200 server using the original baseline's resolved configuration.
# STEPS=2 is a wiring check; STEPS=500 creates a fresh comparison run.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/My_GrowMTP-main}"
BASELINE_RUN="${BASELINE_RUN:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/vn-summarization-growmtp/runs/qwen3-4b-growmtp-full-growmtp-baseline-lora-20261007T144854Z-95705}"
GPU_ID="${GPU_ID:-0}"
STEPS="${STEPS:-2}"
DRY_RUN="${DRY_RUN:-0}"

[[ "$GPU_ID" =~ ^[0-9]+$ && "$STEPS" =~ ^[1-9][0-9]*$ ]] || {
    printf 'ERROR: GPU_ID must be one GPU index; STEPS must be positive.\n' >&2
    exit 2
}
[[ -f "$REPO_ROOT/scripts/common.sh" ]] || {
    printf 'ERROR: repo not found at %s; set REPO_ROOT.\n' "$REPO_ROOT" >&2
    exit 2
}
source "$REPO_ROOT/scripts/common.sh"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONHASHSEED=1
export GROWMTP_LOG_LEVEL=compact

if [[ "$DRY_RUN" != 1 ]]; then
    nvidia-smi --id="$GPU_ID" --query-gpu=index,name,memory.total --format=csv,noheader
fi

# A new run never resumes or writes checkpoints into the source baseline.
RUN_BASE="${RUN_BASE:-$(dirname -- "$BASELINE_RUN")}"
mkdir -p "$RUN_BASE"
AR_RUN_DIR="$(mktemp -d "$RUN_BASE/qwen3-4b-ar-lora-${STEPS}steps-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
export GROWMTP_RUN_DIR="$AR_RUN_DIR"
mkdir -p "$AR_RUN_DIR/config" "$AR_RUN_DIR/logs" "$AR_RUN_DIR/runtime"
AR_RAY_TEMP="$(mktemp -d /tmp/ar-ray-XXXXXX)"

"$GROWMTP_PYTHON" - "$BASELINE_RUN" "$AR_RUN_DIR" "$STEPS" "$AR_RAY_TEMP" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from omegaconf import OmegaConf

baseline, run, steps, ray_temp = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
sources = sorted((baseline / 'config').glob('resolved_config-*.yaml'))
if len(sources) != 1:
    raise SystemExit(f'Expected one original resolved config, found {len(sources)} in {baseline / "config"}. Use the original fresh baseline run on the B200 server.')
source = sources[0]
cfg = OmegaConf.load(source)
original = OmegaConf.to_container(cfg, resolve=True)
cfg = OmegaConf.create(original)
assert cfg.trainer.n_gpus_per_node == 1, 'This recipe matches the one-GPU baseline.'
assert cfg.actor_rollout_ref.rollout.name == 'sglang'
assert cfg.data.seed == 1, 'This recipe matches the seed-1 baseline.'

mtp = cfg.actor_rollout_ref.model.mtp
for key in ('enable', 'enable_train', 'enable_rollout', 'growmtp'):
    mtp[key] = False
mtp.comparison_probe_frequency = 0
mtp.comparison_refresh_fraction = 0
mtp.comparison_log_trajectories = False
mtp.rollout_aux_ce_lambda = 0
# Resolved YAML contains a separate copy of rollout.mtp. Disable it too.
cfg.actor_rollout_ref.rollout.mtp = None

cfg.trainer.total_training_steps = steps
cfg.trainer.resume_mode = 'disable'
cfg.trainer.default_local_dir = str(run / 'checkpoints')
cfg.trainer.rollout_data_dir = str(run / 'artifacts/rollouts')
cfg.trainer.validation_data_dir = str(run / 'artifacts/validation')
cfg.trainer.experiment_name = f'qwen3-4b-ar-lora-{steps}steps'
cfg.global_profiler.save_path = str(run / 'artifacts/profiling')
cfg.ray_kwargs.ray_init._temp_dir = ray_temp
if steps <= 2:
    cfg.trainer.val_before_train = False
    cfg.trainer.final_validation = False
    cfg.trainer.test_freq = -1
    cfg.trainer.save_freq = 1

changed = OmegaConf.to_container(cfg, resolve=True)
def diff(before, after, path=''):
    result = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() | after.keys()):
            result.extend(diff(before.get(key), after.get(key), f'{path}.{key}' if path else key))
    elif before != after:
        result.append({'key': path, 'before': before, 'after': after})
    return result

OmegaConf.save(cfg, run / 'config/ar_no_mtp.yaml', resolve=True)
(run / 'config/changes.json').write_text(json.dumps({
    'source_config': str(source),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'changes': diff(original, changed),
}, ensure_ascii=False, indent=2) + '\n')
print(f'AR run folder: {run}')
print(f'Initial model: {cfg.actor_rollout_ref.model.path}')
print(f'Steps: {steps}; train batch: {cfg.data.train_batch_size}; rollout n: {cfg.actor_rollout_ref.rollout.n}')
print('MTP actor: disabled; MTP rollout: disabled; LoRA/config inherited from baseline.')
PY

if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Config prepared only; training was not launched.\n'
    exit 0
fi

cd "$REPO_ROOT"
"$GROWMTP_PYTHON" "$REPO_ROOT/scripts/run_logged.py" \
    --log-file "$AR_RUN_DIR/logs/training.log" --level compact -- \
    "$GROWMTP_PYTHON" -m verl.trainer.main_ppo \
    --config-path "$AR_RUN_DIR/config" --config-name ar_no_mtp
