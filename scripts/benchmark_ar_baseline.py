#!/usr/bin/env python3
"""Measure an autoregressive SGLang throughput baseline for GrowMTP runs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--response-length", type=int, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-tokens", type=int, default=128)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--tp-per-replica", type=int, default=1)
    parser.add_argument("--memory-fraction", type=float, default=0.6)
    parser.add_argument("--max-running-requests", type=int, default=72)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_prompt_messages(path: Path, prompt_key: str, count: int, seed: int) -> list[list[dict]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if prompt_key not in parquet.schema_arrow.names:
        raise ValueError(f"{path} does not contain prompt column {prompt_key!r}")

    rng = random.Random(seed)
    selected: list[list[dict]] = []
    seen = 0
    for batch in parquet.iter_batches(columns=[prompt_key], batch_size=1024):
        for value in batch.column(0).to_pylist():
            if isinstance(value, str):
                try:
                    decoded = json.loads(value)
                except json.JSONDecodeError:
                    decoded = None
                messages = decoded if isinstance(decoded, list) else [{"role": "user", "content": value}]
            else:
                messages = value
            if not isinstance(messages, list) or not messages:
                continue
            if not all(isinstance(message, dict) and "role" in message for message in messages):
                continue

            seen += 1
            if len(selected) < count:
                selected.append(messages)
            else:
                replacement = rng.randrange(seen)
                if replacement < count:
                    selected[replacement] = messages

    if not selected:
        raise ValueError(f"No chat prompts found in {path}:{prompt_key}")
    return selected


def tokenize_prompts(model_path: str, prompts: list[list[dict]], max_prompt_length: int) -> list[list[int]]:
    from transformers import AutoTokenizer

    from verl.utils.chat_template import apply_chat_template
    from verl.utils.tokenizer import normalize_token_ids

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    result = []
    for messages in prompts:
        prompt_ids = normalize_token_ids(
            apply_chat_template(
                tokenizer,
                messages,
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=True,
            )
        )
        result.append(prompt_ids[-max_prompt_length:])
    if any(not prompt_ids for prompt_ids in result):
        raise ValueError("Chat template produced an empty prompt")
    return result


def completion_tokens(results) -> list[int]:
    if isinstance(results, dict):
        results = [results]
    if not isinstance(results, list) or not results:
        raise RuntimeError(f"Unexpected SGLang output type: {type(results).__name__}")

    counts = []
    for result in results:
        meta = result.get("meta_info", {})
        count = meta.get("completion_tokens")
        if count is None:
            output_ids = result.get("output_ids")
            count = len(output_ids) if output_ids is not None else 0
        counts.append(int(count))
    return counts


def main() -> None:
    args = parse_args()
    if min(args.num_prompts, args.response_length, args.max_prompt_length, args.repeats, args.gpus) <= 0:
        raise ValueError("Prompt count, lengths, repeats, and GPU count must be positive")
    if args.tp_per_replica <= 0 or args.gpus % args.tp_per_replica:
        raise ValueError("GPU count must be divisible by --tp-per-replica")
    if not 0 < args.memory_fraction < 1:
        raise ValueError("--memory-fraction must be between zero and one")

    prompts = load_prompt_messages(Path(args.train_file), args.prompt_key, args.num_prompts, args.seed)
    prompt_ids = tokenize_prompts(args.model_path, prompts, args.max_prompt_length)

    import sglang as sgl
    import torch

    visible_gpus = torch.cuda.device_count()
    if visible_gpus != args.gpus:
        raise RuntimeError(f"Expected {args.gpus} visible GPU(s), found {visible_gpus}")

    # The training config uses TP=1 rollout replicas. A single SGLang DP engine
    # over the same GPUs gives the baseline the same number of independent replicas.
    dp_size = args.gpus // args.tp_per_replica
    engine = sgl.Engine(
        model_path=args.model_path,
        dtype="bfloat16",
        tp_size=args.gpus,
        dp_size=dp_size,
        mem_fraction_static=args.memory_fraction,
        max_running_requests=args.max_running_requests,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        skip_server_warmup=True,
        log_level="error",
    )

    def generate(input_ids: list[list[int]], max_new_tokens: int):
        sampling_params = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": max_new_tokens,
        }
        return engine.generate(input_ids=input_ids, sampling_params=sampling_params)

    try:
        warmup_length = min(args.response_length, args.warmup_tokens)
        warmup_outputs = generate(prompt_ids, warmup_length)
        warmup_count = sum(completion_tokens(warmup_outputs))
        if warmup_count <= 0:
            raise RuntimeError("SGLang warmup generated no completion tokens")

        elapsed_seconds = 0.0
        generated_tokens = 0
        repeat_metrics = []
        for repeat in range(args.repeats):
            started = time.perf_counter()
            outputs = generate(prompt_ids, args.response_length)
            elapsed = time.perf_counter() - started
            counts = completion_tokens(outputs)
            token_count = sum(counts)
            if token_count <= 0 or elapsed <= 0:
                raise RuntimeError(f"AR baseline repeat {repeat + 1} returned no tokens or invalid timing")
            elapsed_seconds += elapsed
            generated_tokens += token_count
            repeat_metrics.append(
                {
                    "repeat": repeat + 1,
                    "elapsed_seconds": elapsed,
                    "generated_tokens": token_count,
                    "tokens_per_second": token_count / elapsed,
                }
            )
            print(
                f"AR baseline repeat {repeat + 1}/{args.repeats}: "
                f"{token_count / elapsed:.2f} tok/s, {token_count} tokens, {elapsed:.2f}s",
                flush=True,
            )
    finally:
        engine.shutdown()

    tokens_per_second = generated_tokens / elapsed_seconds
    result = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "baseline_type": "autoregressive_sglang",
        "model_path": str(Path(args.model_path).resolve()),
        "dataset": str(Path(args.train_file).resolve()),
        "prompt_count": len(prompt_ids),
        "prompt_tokens": sum(map(len, prompt_ids)),
        "max_prompt_length": args.max_prompt_length,
        "response_length": args.response_length,
        "repeats": args.repeats,
        "warmup_tokens": warmup_count,
        "gpus": args.gpus,
        "tp_per_replica": args.tp_per_replica,
        "dp_replicas": dp_size,
        "gpu_memory_fraction": args.memory_fraction,
        "max_running_requests": args.max_running_requests,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "elapsed_seconds": elapsed_seconds,
        "generated_tokens": generated_tokens,
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_per_gpu": tokens_per_second / args.gpus,
        "ms_per_token": elapsed_seconds * 1000.0 / generated_tokens,
        "repeat_metrics": repeat_metrics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(output_path)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
