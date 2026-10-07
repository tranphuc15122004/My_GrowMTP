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
choose a larger limit, prepare into a new versioned `DATA_DIR`, and use that same
limit for both runs. No training starts with a prompt that GrowMTP would trim.

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
  RUN_MODE=smoke bash scripts/run_vn_summarization_growmtp.sh --gpuid 2,3

DATA_DIR="$DATA_DIR" BASE_MODEL="$BASE_MODEL" GROWMTP_SEED=1 \
  RUN_MODE=smoke MTP_AUX_CE_LAMBDA=0.001 \
  bash scripts/run_vn_summarization_growmtp.sh --gpuid 2,3

DATA_DIR="$DATA_DIR" BASE_MODEL="$BASE_MODEL" GROWMTP_SEED=1 \
  RUN_MODE=pilot TRAIN_STEPS=30 \
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
The wrapper assigns distinct run labels and always creates a fresh run folder.

```bash
DATA_DIR="$DATA_DIR" BASE_MODEL="$BASE_MODEL" GROWMTP_SEED=1 \
  RUN_MODE=full TRAIN_STEPS=500 SAVE_FREQ=50 SAVE_GENERATIONS=1 \
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

For an initial 500-step Idea experiment in the background on one B200, after
the prompt audit succeeds, keep the smoke coefficient at 0.001. Four smoke
steps establish runtime compatibility; they do not calibrate the coefficient
or establish a speedup. The command below starts a fresh run and retains the
default target LR (1e-6), draft LR (3e-4), 10-step draft warmup, and LoRA rank
16/alpha 32:

```bash
LOG_DIR=/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/vn-summarization-growmtp
mkdir -p "$LOG_DIR"
FULL_LOG="$LOG_DIR/idea-full-gpu0-$(date -u +%Y%m%dT%H%M%SZ).log"

nohup env DATA_DIR="$DATA_DIR" BASE_MODEL="$BASE_MODEL" \
  GROWMTP_SEED="$GROWMTP_SEED" MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
  TRAIN_FILE="$DATA_DIR/train.parquet" VAL_FILE="$DATA_DIR/validation.parquet" \
  GROWMTP_DATA_MANIFEST="$DATA_DIR/manifest.json" \
  RUN_MODE=full RUN_ACTION=fresh RUN_DIR= RUN_OUTPUT_DIR= PREPARED_MODEL_DIR= \
  TRAIN_STEPS=500 TRAIN_BATCH_SIZE=4 ROLLOUT_N=4 RESPONSE_LENGTH=512 \
  PPO_MINI_BATCH_SIZE=4 AGENT_LOOP_WORKERS=4 \
  MTP_AUX_CE_LAMBDA=0.001 MTP_AUX_ADVANTAGE_CLIP=2 \
  SAVE_FREQ=50 SAVE_GENERATIONS=1 AR_BASELINE_AUTO=0 \
  VAL_BEFORE_TRAIN=1 FINAL_VALIDATION=1 TEST_FREQ=-1 VAL_MAX_SAMPLES=128 VAL_SAMPLES=4 \
  bash scripts/run_vn_summarization_growmtp.sh --gpuid 0 \
  > "$FULL_LOG" 2>&1 < /dev/null &

echo "$!" > "$FULL_LOG.pid"
echo "Log: $FULL_LOG"
tail -f "$FULL_LOG"
```

Use the same configuration with `MTP_AUX_CE_LAMBDA=0` for the GrowMTP baseline.
