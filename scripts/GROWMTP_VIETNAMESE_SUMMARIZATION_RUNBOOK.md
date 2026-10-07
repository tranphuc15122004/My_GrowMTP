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
  RUN_MODE=full TRAIN_STEPS=500 SAVE_GENERATIONS=1 \
  bash scripts/run_vn_summarization_growmtp.sh --gpuid 2,3

IDEA_LAMBDA=0.001  # replace with the coefficient calibrated from the pilot
DATA_DIR="$DATA_DIR" BASE_MODEL="$BASE_MODEL" GROWMTP_SEED=1 \
  RUN_MODE=full TRAIN_STEPS=500 SAVE_GENERATIONS=1 MTP_AUX_CE_LAMBDA="$IDEA_LAMBDA" \
  bash scripts/run_vn_summarization_growmtp.sh --gpuid 2,3
```

Compare validation ROUGE-L, end-to-end and per-step time, throughput, MTP
acceptance length, response clipping, and auxiliary CE activity. ROUGE-L is the
training reward and does not alone measure factuality, so inspect a sample of
the final validation summaries before drawing a quality conclusion.
