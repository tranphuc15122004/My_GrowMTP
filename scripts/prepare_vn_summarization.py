"""Convert Vietnamese text/summary JSONL to GrowMTP's veRL Parquet schema."""

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


SCHEMA = pa.schema(
    [
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("data_source", pa.string()),
        ("ability", pa.string()),
        ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
        ("extra_info", pa.struct([("id", pa.string())])),
    ]
)


def convert_record(record: dict, line_number: int) -> dict:
    if not isinstance(record, dict):
        raise ValueError(f"Line {line_number}: expected a JSON object")
    for field in ("id", "text", "summary"):
        if field not in record or not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"Line {line_number}: {field} must be a non-empty string")
    article = record["text"].strip()
    return {
        "prompt": [{
            "role": "user",
            "content": "Tóm tắt văn bản sau bằng tiếng Việt, ngắn gọn và đúng sự thật. "
                       "Chỉ viết phần tóm tắt, không thêm nhận xét.\n\nVăn bản:\n"
                       f"{article}\n\nTóm tắt:",
        }],
        "data_source": "vn_summarization",
        "ability": "summarization",
        "reward_model": {"style": "rule", "ground_truth": record["summary"].strip()},
        "extra_info": {"id": record["id"]},
    }


def load_rows(source: Path) -> list[dict]:
    rows = []
    with source.open("rb") as stream, tqdm(total=source.stat().st_size, unit="B", unit_scale=True,
                                         desc="Reading JSONL") as progress:
        for line_number, raw in enumerate(stream, 1):
            progress.update(len(raw))
            line = raw.decode("utf-8-sig").strip()
            if not line:
                continue
            if line_number == 1 and line.startswith("Path: "):
                # data/sample.txt has a source-path note; the real JSONL does not.
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON: {exc}") from exc
            rows.append(convert_record(record, line_number))
    if len(rows) < 2:
        raise ValueError("At least two valid records are required for train/validation splitting")
    return rows


def split_rows(rows: list[dict], validation_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    groups = {}
    for row in rows:
        article = row["prompt"][0]["content"]
        key = hashlib.sha256(article.encode("utf-8")).hexdigest()
        groups.setdefault(key, []).append(row)
    if len(groups) < 2:
        raise ValueError("At least two distinct articles are required for a leakage-free split")
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target_validation = max(1, min(len(rows) - 1, round(len(rows) * validation_fraction)))
    train, validation = [], []
    for key in keys:
        group = groups[key]
        if len(validation) < target_validation and len(validation) + len(group) < len(rows):
            validation.extend(group)
        else:
            train.extend(group)
    if not train or not validation:
        raise ValueError("Could not create non-empty train and validation splits")
    return train, validation


def measure_prompt_lengths(rows: list[dict], tokenizer) -> list[int]:
    """Count the complete chat prompt, including the generation template."""
    lengths = []
    for row in tqdm(rows, desc="Auditing prompt tokens"):
        token_ids = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        lengths.append(len(token_ids))
    if not lengths:
        raise ValueError("Cannot audit an empty prompt dataset")
    return lengths


def audit_lengths(lengths: list[int], max_prompt_length: int) -> dict:
    if max_prompt_length < 1:
        raise ValueError("max_prompt_length must be a positive integer")
    if not lengths:
        raise ValueError("Cannot audit an empty prompt dataset")
    ordered = sorted(lengths)
    maximum = ordered[-1]
    audit = {
        "prompt_count": len(lengths),
        "max_prompt_tokens": maximum,
        "p95_prompt_tokens": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "max_prompt_length": max_prompt_length,
    }
    if maximum > max_prompt_length:
        raise ValueError(
            f"Prompt length {maximum} exceeds max_prompt_length={max_prompt_length}; "
            f"p95={audit['p95_prompt_tokens']}, "
            f"over_limit={sum(length > max_prompt_length for length in lengths)}/{len(lengths)}; "
            "increase the limit or prepare with --drop-overlong-prompts before training"
        )
    return audit


def audit_prompt_lengths(rows: list[dict], tokenizer, max_prompt_length: int) -> dict:
    """Measure complete Qwen chat prompts and reject inputs GrowMTP would truncate."""
    return audit_lengths(measure_prompt_lengths(rows, tokenizer), max_prompt_length)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw JSONL with id/text/summary")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for Parquet files")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("GROWMTP_SEED", "1")))
    parser.add_argument("--tokenizer", help="Qwen3 tokenizer path for the required server prompt audit")
    parser.add_argument(
        "--max-prompt-length", type=int, default=int(os.environ.get("MAX_PROMPT_LENGTH", "4096"))
    )
    parser.add_argument(
        "--drop-overlong-prompts", action="store_true",
        help="Drop complete prompts above --max-prompt-length after splitting; requires --tokenizer",
    )
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.seed < 0:
        parser.error("--seed must be a non-negative integer")
    if args.max_prompt_length < 1:
        parser.error("--max-prompt-length must be positive")
    if args.drop_overlong_prompts and not args.tokenizer:
        parser.error("--drop-overlong-prompts requires --tokenizer")
    if not args.input.is_file():
        parser.error(f"Input JSONL not found: {args.input}")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")

    rows = load_rows(args.input)
    print(f"Loaded {len(rows):,} records; splitting train/validation with seed={args.seed}...", flush=True)
    train, validation = split_rows(rows, args.validation_fraction, args.seed)
    print(f"Split ready: {len(train):,} train, {len(validation):,} validation before filtering.", flush=True)
    prompt_audit = None
    prompt_filter = None
    if args.tokenizer:
        print("Importing Transformers tokenizer dependencies...", flush=True)
        from transformers import AutoTokenizer

        print(f"Loading tokenizer: {args.tokenizer}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
        print(f"Tokenizer ready; auditing complete prompts (limit={args.max_prompt_length:,} tokens)...", flush=True)
        if args.drop_overlong_prompts:
            # Filter after splitting so surviving records keep their original partition/order.
            train_count, validation_count = len(train), len(validation)
            lengths = measure_prompt_lengths(train + validation, tokenizer)
            train = [row for row, length in zip(train, lengths[:train_count])
                     if length <= args.max_prompt_length]
            validation = [row for row, length in zip(validation, lengths[train_count:])
                          if length <= args.max_prompt_length]
            if not train or not validation:
                parser.error("Prompt filtering must leave non-empty train and validation splits")
            kept_lengths = [length for length in lengths if length <= args.max_prompt_length]
            prompt_audit = audit_lengths(kept_lengths, args.max_prompt_length)
            prompt_filter = {
                "max_prompt_length": args.max_prompt_length,
                "source_rows": len(rows),
                "kept_rows": len(train) + len(validation),
                "dropped_rows": len(rows) - len(train) - len(validation),
                "dropped_train_rows": train_count - len(train),
                "dropped_validation_rows": validation_count - len(validation),
                "max_prompt_tokens_before": max(lengths),
            }
            print(
                f"Dropped {prompt_filter['dropped_rows']:,}/{len(rows):,} overlong prompts "
                f"(train={prompt_filter['dropped_train_rows']:,}, "
                f"validation={prompt_filter['dropped_validation_rows']:,}); "
                f"kept max_prompt_tokens={prompt_audit['max_prompt_tokens']:,}/{args.max_prompt_length:,}",
                flush=True,
            )
        else:
            prompt_audit = audit_prompt_lengths(rows, tokenizer, args.max_prompt_length)
        prompt_audit["tokenizer"] = str(args.tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Writing train.parquet ({len(train):,} records)...", flush=True)
    pq.write_table(pa.Table.from_pylist(train, schema=SCHEMA), args.output_dir / "train.parquet")
    print(f"Writing validation.parquet ({len(validation):,} records)...", flush=True)
    pq.write_table(pa.Table.from_pylist(validation, schema=SCHEMA), args.output_dir / "validation.parquet")
    manifest = {
        "source": str(args.input.resolve()), "seed": args.seed,
        "train_rows": len(train), "validation_rows": len(validation),
        "validation_fraction_requested": args.validation_fraction,
        "unique_articles": len({r["prompt"][0]["content"] for r in train + validation}),
        "prompt_audit": prompt_audit,
    }
    if prompt_filter is not None:
        manifest["prompt_filter"] = prompt_filter
    print("Writing manifest.json...", flush=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(train):,} train and {len(validation):,} validation records in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
