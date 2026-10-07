# GrowMTP LoRA for Vietnamese summarization on B200

This runbook prepares the JSONL source listed in `data/sample.txt` and compares
plain GrowMTP with rollout-advantage auxiliary CE. The real source starts with a
JSON object on line 1; the `Path:` note appears only in `data/sample.txt`.

## Prepare data on the server

Activate the same Python environment used for training, then set the paths and
the shared experiment seed:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/My_GrowMTP-main
export GROWMTP_PYTHON="$(python -c 'import sys; print(sys.executable)')"

RAW=/workspace/storage-shared/nlp/dungdx4/bien_projects/LLM2Seq/src/eviseq_new/datasets/50k/train_clean.jsonl
BASE_MODEL=/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B
DATA_DIR=/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/datasets/vn_summarization/seed1-v1
GROWMTP_SEED=1
MAX_PROMPT_LENGTH=4096

python scripts/prepare_vn_summarization.py \
  --input "$RAW" \
  --output-dir "$DATA_DIR" \
  --seed "$GROWMTP_SEED" \
  --tokenizer "$BASE_MODEL" \
  --max-prompt-length "$MAX_PROMPT_LENGTH"
```

Preparation creates `train.parquet`, `validation.parquet`, and `manifest.json`.
The manifest records the split seed and the maximum complete prompt length
measured with the Qwen3 tokenizer and chat template. The script refuses an
existing output directory. If the audit reports a prompt above 4096 tokens,
either choose a larger limit or drop overlong records with the option below.
Use the same dataset and prompt limit for both runs. No training starts with a
prompt that GrowMTP would trim.

To keep the 4096-token limit on one B200, prepare a filtered dataset into a new
directory. This removes outliers by complete chat-prompt token length; it does
not truncate articles or alter reference summaries:

```bash
DATA_DIR=/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/datasets/vn_summarization/seed1-v2-filter4096

"$GROWMTP_PYTHON" scripts/prepare_vn_summarization.py \
  --input "$RAW" \
  --output-dir "$DATA_DIR" \
  --validation-fraction 0.1 \
  --seed "$GROWMTP_SEED" \
  --tokenizer "$BASE_MODEL" \
  --max-prompt-length 4096 \
  --drop-overlong-prompts
```

Filtering happens after the original seeded split: surviving rows retain their
train/validation partition and order when the raw source, seed, and validation
fraction are unchanged. Prompts exactly at the limit are retained. The script
prints removed counts for both splits and records them in `manifest.json` under
`prompt_filter`; `prompt_audit` describes only retained prompts. It rejects an
empty resulting split before writing any output. The old dataset stays intact.
When launching training, set `TRAIN_FILE`, `VAL_FILE`, and
`GROWMTP_DATA_MANIFEST` to this new directory as well as `DATA_DIR`.

If an older manifest reports `max_prompt_tokens=2`, its audit counted the fields
of the Transformers `BatchEncoding` rather than the token IDs. After updating
the preparation and validation scripts, recount the existing Parquet prompts:

```bash
python scripts/validate_vn_summarization_data.py \
  --manifest "$DATA_DIR/manifest.json" \
  --seed "$GROWMTP_SEED" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  --refresh-prompt-audit --tokenizer "$BASE_MODEL"
```

This updates only the audit metadata after a successful check. If a prompt is
over the limit, it reports the maximum, p95, and number of overlong prompts and
leaves the existing files intact. Choose the prompt limit before launching full
training; the Parquet split and seed stay the same.

## Smoke and pilot

The launcher selects the active Python environment automatically and writes
logs and checkpoints below `outputs/vn-summarization-growmtp/runs/`. Choose any
available B200 pair with `--gpuid`, for example `2,3`.

```bash
DATA_DIR="$DATA_DIR" BASE_MODEL="$BASE_MODEL" GROWMTP_SEED=1 \
  RUN_MODE=smoke MTP_AUX_CE_LAMBDA=0 \
  bash scripts/run_vn_summarization_growmtp.sh --gpuid 2,3

DATA_DIR="$DATA_DIR" BASE_MODEL="$BASE_MODEL" GROWMTP_SEED=1 \
  RUN_MODE=smoke MTP_AUX_CE_LAMBDA=0.001 \
  bash scripts/run_vn_summarization_growmtp.sh --gpuid 2,3

DATA_DIR="$DATA_DIR" BASE_MODEL="$BASE_MODEL" GROWMTP_SEED=1 \
  RUN_MODE=pilot TRAIN_STEPS=30 MTP_AUX_CE_LAMBDA=0 \
  bash scripts/run_vn_summarization_growmtp.sh --gpuid 2,3

DATA_DIR="$DATA_DIR" BASE_MODEL="$BASE_MODEL" GROWMTP_SEED=1 \
  RUN_MODE=pilot TRAIN_STEPS=30 MTP_AUX_CE_LAMBDA=0.001 \
  bash scripts/run_vn_summarization_growmtp.sh --gpuid 2,3
```

The seed must match the manifest. It controls the split, MTP head initialization,
LoRA initialization, dataloader order, and SGLang random seed. The launcher
records it in each run and restores it when resuming. Both arms use LoRA rank 16,
alpha 32, `q_proj`/`v_proj`, and merged adapter weights for EAGLE rollout.

Before full runs, check that reward values vary, advantages are nonzero, and
the Idea run reports positive auxiliary CE cycles. The `0.001` coefficient is
for smoke/pilot wiring only. For the full Idea run, use the coefficient selected
from a positive-advantage batch so the weighted CE gradient is about 10% of the
DCA gradient.

## Full comparison

Use the same `DATA_DIR`, seed, GPU pair, and configuration in both commands.
Run the plain GrowMTP baseline first. Both runs should start fresh from the same
base model; do not initialize the Idea run from the trained baseline checkpoint.
The wrapper assigns distinct run labels and creates a fresh run folder by default.

```bash
DATA_DIR="$DATA_DIR" BASE_MODEL="$BASE_MODEL" GROWMTP_SEED=1 \
  RUN_MODE=full TRAIN_STEPS=500 SAVE_FREQ=50 SAVE_GENERATIONS=1 MTP_AUX_CE_LAMBDA=0 \
  bash scripts/run_vn_summarization_growmtp.sh --gpuid 2,3

IDEA_LAMBDA=0.001  # replace with the coefficient calibrated from the pilot
DATA_DIR="$DATA_DIR" BASE_MODEL="$BASE_MODEL" GROWMTP_SEED=1 \
  RUN_MODE=full TRAIN_STEPS=500 SAVE_FREQ=50 SAVE_GENERATIONS=1 MTP_AUX_CE_LAMBDA="$IDEA_LAMBDA" \
  bash scripts/run_vn_summarization_growmtp.sh --gpuid 2,3
```

Compare validation ROUGE-L, end-to-end and per-step time, throughput, MTP
acceptance length, response clipping, and auxiliary CE activity. ROUGE-L is the
training reward and does not alone measure factuality, so inspect a sample of
the final validation summaries before drawing a quality conclusion.

For the initial 500-step GrowMTP baseline in the background on one B200, run
the command below after the prompt audit succeeds. Auxiliary CE, policy-shift
probes, and refresh are disabled; native GrowMTP DCA/VGM training remains enabled.
The command starts a fresh run and retains the default target LR (1e-6), draft
LR (3e-4), 10-step draft warmup, and LoRA rank 16/alpha 32:

`RUN_MODE=full` selects the complete 500-step schedule, not full-parameter
fine-tuning. The target backbone is frozen by PEFT; only its LoRA matrices and
the native GrowMTP MTP head are trained. Adapter merging for rollout does not
unfreeze the target backbone.

For a conservative direct launch on one dedicated 180 GB B200, keep the
comparison batch at four prompts and four rollouts per prompt. Use eight
agent-loop workers, an 8192-token dynamic training budget, and a SGLang memory
fraction of 0.5. Budget roughly 90 GB for inference weights/KV cache, 25-35 GB
for resident training/reference weights and optimizer state, and 15-25 GB for
temporary activations, CUDA buffers, and allocator overhead. The resulting
130-150 GB planning estimate leaves room below device capacity; it is not a
measured peak or a hard limit on combined process memory. The console GPU
figure measures the actor process, not total device memory. These settings
keep the existing gradient checkpointing and avoid changing the algorithm.

```bash
LOG_DIR=/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/vn-summarization-growmtp
mkdir -p "$LOG_DIR"
FULL_LOG="$LOG_DIR/growmtp-baseline-full-gpu0-$(date -u +%Y%m%dT%H%M%SZ).log"

nohup env DATA_DIR="$DATA_DIR" BASE_MODEL="$BASE_MODEL" \
  RUN_BASE_DIR="$LOG_DIR/runs" \
  GROWMTP_SEED="$GROWMTP_SEED" MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
  TRAIN_FILE="$DATA_DIR/train.parquet" VAL_FILE="$DATA_DIR/validation.parquet" \
  GROWMTP_DATA_MANIFEST="$DATA_DIR/manifest.json" \
  RUN_MODE=full RUN_ACTION=fresh RUN_LABEL=growmtp-baseline \
  RUN_DIR= RUN_OUTPUT_DIR= PREPARED_MODEL_DIR= \
  TRAIN_STEPS=500 TRAIN_BATCH_SIZE=4 ROLLOUT_N=4 RESPONSE_LENGTH=512 \
  PPO_MINI_BATCH_SIZE=4 AGENT_LOOP_WORKERS=8 \
  MAX_TOKEN_LEN_PER_GPU=8192 ROLLOUT_GPU_MEMORY_UTILIZATION=0.5 \
  LORA_RANK=16 LORA_ALPHA=32 TARGET_MODULES_JSON='["q_proj","v_proj"]' \
  MTP_AUX_CE_LAMBDA=0 MTP_PROBE_FREQ=0 MTP_REFRESH_FRACTION=0 \
  SAVE_FREQ=50 SAVE_GENERATIONS=1 AR_BASELINE_AUTO=0 \
  VAL_BEFORE_TRAIN=1 FINAL_VALIDATION=1 TEST_FREQ=-1 VAL_MAX_SAMPLES=128 VAL_SAMPLES=4 \
  bash scripts/run_vn_summarization_growmtp.sh --gpuid 0 \
  > "$FULL_LOG" 2>&1 < /dev/null &

echo "$!" > "$FULL_LOG.pid"
echo "Log: $FULL_LOG"
tail -F "$FULL_LOG"
```

After the baseline finishes, use the same configuration for a fresh Idea run,
changing the run label and log filename and enabling auxiliary CE with the
chosen coefficient. Keep the dataset, seed, GPU, batch sizes, sequence lengths,
checkpoint frequency, and validation settings identical. Four smoke steps do
not calibrate the CE coefficient or establish a speedup.
