#!/usr/bin/env python3
"""Validate the Vietnamese summarization data manifest before a GrowMTP run."""

import argparse
import json
from pathlib import Path


def validate_manifest(path: Path, expected_seed: int, max_prompt_length: int) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read data manifest {path}: {exc}") from exc
    if manifest.get("seed") != expected_seed:
        raise ValueError(
            f"Data split seed {manifest.get('seed')!r} does not match GROWMTP_SEED={expected_seed}"
        )
    audit = manifest.get("prompt_audit")
    if not isinstance(audit, dict):
        raise ValueError("Prompt audit is missing; prepare data with --tokenizer before training")
    if not isinstance(audit.get("max_prompt_tokens"), int) or audit["max_prompt_tokens"] < 1:
        raise ValueError("Prompt audit has no valid max_prompt_tokens value")
    if audit["max_prompt_tokens"] > max_prompt_length:
        raise ValueError(
            f"Audited prompt length {audit['max_prompt_tokens']} exceeds "
            f"MAX_PROMPT_LENGTH={max_prompt_length}; rerun preparation with a larger limit"
        )
    if not isinstance(audit.get("tokenizer"), str) or not audit["tokenizer"]:
        raise ValueError("Prompt audit does not record the tokenizer path")
    for name in ("train_rows", "validation_rows"):
        if not isinstance(manifest.get(name), int) or manifest[name] < 1:
            raise ValueError(f"Manifest has no positive {name} count")
    return manifest


def refresh_prompt_audit(path: Path, manifest: dict, tokenizer_path: str, max_prompt_length: int) -> dict:
    """Recount existing Parquet prompts and atomically replace only manifest metadata."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from prepare_vn_summarization import audit_prompt_lengths

    rows = []
    for filename, count_key in (("train.parquet", "train_rows"), ("validation.parquet", "validation_rows")):
        split_rows = pq.read_table(path.parent / filename, columns=["prompt"]).to_pylist()
        if len(split_rows) != manifest[count_key]:
            raise ValueError(f"{filename} row count does not match {count_key} in the manifest")
        rows.extend(split_rows)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    audit = audit_prompt_lengths(rows, tokenizer, max_prompt_length)
    audit["tokenizer"] = tokenizer_path
    manifest["prompt_audit"] = audit
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-prompt-length", type=int, required=True)
    parser.add_argument(
        "--refresh-prompt-audit", action="store_true",
        help="Recount train.parquet and validation.parquet beside the manifest; update audit metadata",
    )
    parser.add_argument("--tokenizer", help="Tokenizer path required with --refresh-prompt-audit")
    args = parser.parse_args()
    if args.max_prompt_length < 1:
        parser.error("--max-prompt-length must be positive")
    if args.refresh_prompt_audit and not args.tokenizer:
        parser.error("--refresh-prompt-audit requires --tokenizer")
    try:
        manifest = validate_manifest(args.manifest, args.seed, args.max_prompt_length)
        if args.refresh_prompt_audit:
            manifest = refresh_prompt_audit(
                args.manifest, manifest, args.tokenizer, args.max_prompt_length
            )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    audit = manifest["prompt_audit"]
    print(
        "Vietnamese summarization data OK: "
        f"seed={args.seed}, train={manifest['train_rows']:,}, "
        f"validation={manifest['validation_rows']:,}, "
        f"max_prompt_tokens={audit['max_prompt_tokens']:,}/{args.max_prompt_length:,}"
    )


if __name__ == "__main__":
    main()
