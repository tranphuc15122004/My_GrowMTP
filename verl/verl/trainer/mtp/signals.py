"""Tensor-batch transport for ragged verification cycles and committed-prefix inputs."""

import torch

FIELDS = (
    "hidden",
    "draft_tokens",
    "topk_val",
    "topk_idx",
    "position",
    "accept_len",
    "prefix_hidden",
    "prefix_ids",
)
TARGET_FIELDS = ("target_tokens", "target_mask")


def logprob_inputs(batch):
    """Select inference fields without copying or modifying the training batch."""
    tensor_keys = list(batch.batch.keys()) if batch.batch is not None else None
    keep_tensor_keys = (
        [key for key in tensor_keys if not (isinstance(key, str) and key.startswith("mtp_"))]
        if tensor_keys is not None else None
    )
    keep_non_tensor_keys = [
        key for key in batch.non_tensor_batch
        if not (isinstance(key, str) and key.startswith("mtp_"))
    ]
    if (keep_tensor_keys == tensor_keys
            and len(keep_non_tensor_keys) == len(batch.non_tensor_batch)):
        return batch
    return batch.select(batch_keys=keep_tensor_keys, non_tensor_batch_keys=keep_non_tensor_keys)


def _prefill_pairs(signal, hidden_size):
    blocks = [signal.get(name) for name in ("prefill_hidden", "prefill_tokens", "prefill_position")]
    if all(block is None for block in blocks):
        return {}
    if any(block is None for block in blocks) or len({len(block) for block in blocks}) != 1:
        raise ValueError("Re-prefill history fields are incomplete")
    pairs = {}
    for hidden, tokens, positions in zip(*blocks):
        hidden, tokens, positions = map(torch.as_tensor, (hidden, tokens, positions))
        if (hidden.ndim != 2 or tokens.ndim != 1 or positions.ndim != 1
                or hidden.shape != (tokens.numel(), hidden_size)
                or tokens.numel() != positions.numel()
                or positions.dtype != torch.int64 or (positions < 0).any()):
            raise ValueError("Re-prefill hidden states, tokens and positions are misaligned")
        for i, position in enumerate(positions.tolist()):
            if position in pairs:
                raise ValueError(f"Duplicate re-prefill position {position}")
            pairs[position] = hidden[i], tokens[i]
    return pairs


def _prefix_inputs(signal):
    if signal.get("prompt_hidden") is None or signal.get("prompt_ids") is None:
        raise ValueError("GrowMTP requires complete prompt inputs; disable radix caching")
    prompt_hidden = torch.as_tensor(signal["prompt_hidden"])
    prompt_ids = torch.as_tensor(signal["prompt_ids"])
    positions = torch.as_tensor(signal["position"], dtype=torch.int64)
    if (prompt_hidden.ndim != 2 or prompt_ids.ndim != 1
            or prompt_hidden.shape[0] != prompt_ids.numel() or not prompt_ids.numel()):
        raise ValueError("Prompt hidden states and token IDs must cover the same positions")
    if (positions.ndim != 1 or not positions.numel()
            or int(positions[0]) != prompt_ids.numel() - 1
            or (positions[1:] <= positions[:-1]).any()):
        raise ValueError("Cycle positions must start at the final prompt position and increase")
    accepted_hidden = signal.get("accepted_hidden") or []
    accepted_tokens = signal.get("accepted_tokens") or []
    n = positions.numel()
    if len(accepted_hidden) != len(accepted_tokens) or not n - 1 <= len(accepted_hidden) <= n:
        raise ValueError("Committed history must contain one accepted block between each pair of cycles")
    prefill_pairs = _prefill_pairs(signal, prompt_hidden.shape[1])
    hidden_rows, token_rows = [prompt_hidden], [prompt_ids]
    for i in range(n - 1):
        hidden = torch.as_tensor(accepted_hidden[i])
        tokens = torch.as_tensor(accepted_tokens[i])
        length = int(positions[i + 1] - positions[i])
        accepted_length = int(signal["accept_len"][i])
        if (hidden.ndim != 2 or tokens.ndim != 1
                or not 1 <= accepted_length <= length
                or hidden.shape != (accepted_length, prompt_hidden.shape[1])
                or tokens.numel() != accepted_length):
            raise ValueError(f"Accepted block {i} does not cover its committed positions")
        hidden_rows.append(hidden)
        token_rows.append(tokens)
        if accepted_length < length:
            missing = range(int(positions[i]) + accepted_length + 1, int(positions[i + 1]) + 1)
            if any(position not in prefill_pairs for position in missing):
                raise ValueError(f"Re-prefill history is incomplete before cycle {i + 1}")
            hidden_rows.append(torch.stack([prefill_pairs[p][0] for p in missing]).to(hidden))
            token_rows.append(torch.stack([prefill_pairs[p][1] for p in missing]).to(tokens))
    prefix_hidden = torch.cat(hidden_rows, dim=0)
    prefix_ids = torch.cat(token_rows, dim=0)
    for i, position in enumerate(positions.tolist()):
        seed_hidden = torch.as_tensor(signal["hidden"][i]).to(prefix_hidden)
        seed_token = int(torch.as_tensor(signal["draft_tokens"][i])[0])
        if not torch.equal(prefix_hidden[position], seed_hidden) or int(prefix_ids[position]) != seed_token:
            raise ValueError(f"Committed prefix does not match cycle {i}")
    return prefix_hidden, prefix_ids


def _rollout_targets(signal, response_ids, response_mask):
    """Map verifier depths to target-emitted response tokens on the committed path."""
    positions = torch.as_tensor(signal["position"], dtype=torch.int64).cpu()
    accept_len = torch.as_tensor(signal["accept_len"], dtype=torch.int64).cpu()
    accepted_tokens = signal.get("accepted_tokens") or []
    prompt_len = torch.as_tensor(signal["prompt_ids"]).numel()
    depth = torch.as_tensor(signal["topk_idx"][0]).shape[0]
    response_ids = torch.as_tensor(response_ids, dtype=torch.int64).cpu()
    response_mask = torch.as_tensor(response_mask, dtype=torch.bool).cpu()
    if response_ids.shape != response_mask.shape:
        raise ValueError("Response tokens and mask must have matching shapes")

    target_tokens = torch.zeros((positions.numel(), depth), dtype=torch.int64)
    target_mask = torch.zeros((positions.numel(), depth), dtype=torch.bool)
    for cycle, position in enumerate(positions.tolist()):
        valid_depth = min(int(accept_len[cycle]), depth)
        emitted = (torch.as_tensor(accepted_tokens[cycle], dtype=torch.int64)
                   if cycle < len(accepted_tokens) else None)
        for offset in range(valid_depth):
            # The recorded prompt is left-shifted: its final token is the
            # target's first rollout token, which seeds the draft. Verifier
            # depth zero predicts the following response token.
            response_index = position + offset + 2 - prompt_len
            if response_index < 0 or response_index >= response_ids.numel():
                continue
            if response_mask[response_index]:
                if emitted is not None:
                    if offset >= emitted.numel() or int(emitted[offset]) != int(response_ids[response_index]):
                        raise ValueError("Rollout response tokens disagree with verification history")
                target_tokens[cycle, offset] = response_ids[response_index]
                target_mask[cycle, offset] = True
    return target_tokens, target_mask


def pack_signals(batch, records, *, include_target_tokens=False):
    """Remove Python-object payloads at the rollout boundary; keep one nested tensor per field."""
    present = [s for s in records if s and len(s.get("position", []))]
    if not present:
        batch["mtp_num_cycles"] = torch.zeros(len(records), dtype=torch.int64)
        return
    prefixes = {id(signal): _prefix_inputs(signal) for signal in present}
    fields = FIELDS + (TARGET_FIELDS if include_target_tokens else ())
    rows = {key: [] for key in fields}
    target_rows = None
    if include_target_tokens:
        prompts = batch["prompts"].detach().cpu()
        prompt_mask = batch["attention_mask"][:, :prompts.shape[1]].detach().cpu()
        responses = batch["responses"].detach().cpu().unbind(0)
        response_masks = batch["response_mask"].detach().cpu().unbind(0)
        if (len(prompts) != len(records) or len(responses) != len(records)
                or len(response_masks) != len(records)):
            raise ValueError("Rollout records and response rows are misaligned")
        for i, signal in enumerate(records):
            if signal and len(signal.get("position", [])):
                signal_prompt = torch.as_tensor(signal["prompt_ids"], dtype=torch.int64).cpu()
                rollout_prompt = prompts[i][prompt_mask[i].bool()]
                if (signal_prompt.numel() != rollout_prompt.numel() or not signal_prompt.numel()
                        or not bool(response_masks[i][0])
                        or not torch.equal(signal_prompt[:-1], rollout_prompt[1:])
                        or int(signal_prompt[-1]) != int(responses[i][0])):
                    raise ValueError("Auxiliary CE requires the shifted prompt and rollout seed to match")
        target_rows = [
            _rollout_targets(signal, responses[i], response_masks[i])
            if signal and len(signal.get("position", [])) else None
            for i, signal in enumerate(records)
        ]
        target_prototype = next(row for row in target_rows if row is not None)
    counts = []
    prototype = present[0]
    for row_index, signal in enumerate(records):
        n = len(signal["position"]) if signal else 0
        counts.append(n)
        source = signal if n else prototype
        for key in fields:
            if key in TARGET_FIELDS:
                value = target_rows[row_index][0 if key == "target_tokens" else 1] if n else None
                if not n:
                    value = torch.zeros_like(target_prototype[0 if key == "target_tokens" else 1][:1])
                rows[key].append(value.detach().cpu())
                continue
            if key.startswith("prefix_"):
                value = prefixes[id(source)][0 if key == "prefix_hidden" else 1]
            else:
                if source.get(key) is None or len(source[key]) != len(source["position"]):
                    raise ValueError(f"Missing or misaligned verification field: {key}")
                value = torch.stack([torch.as_tensor(x) for x in source[key]])
            dtype = (
                torch.bfloat16 if key in ("hidden", "prefix_hidden", "topk_val") else torch.int64
            )
            value = value.detach().to(device="cpu", dtype=dtype)
            rows[key].append(value if n else torch.zeros_like(value[:1]))
    for key, values in rows.items():
        batch["mtp_" + key] = torch.nested.nested_tensor(values, layout=torch.jagged)
    batch["mtp_num_cycles"] = torch.tensor(counts, dtype=torch.int64)


def unpack_signals(data):
    if "mtp_num_cycles" not in data:
        return []
    counts = data["mtp_num_cycles"].detach().cpu().tolist()
    if not any(counts):
        return []
    fields = FIELDS + (TARGET_FIELDS if "mtp_target_tokens" in data else ())
    rows = {key: data["mtp_" + key].unbind() for key in fields
            if key not in ("position", "accept_len")}
    for key in ("position", "accept_len"):
        rows[key] = data["mtp_" + key].detach().cpu().unbind()
    result = []
    for i, n in enumerate(counts):
        if n:
            row = {key: rows[key][i] for key in fields}
            row["batch_index"] = i
            if row["position"].numel() != n or row["accept_len"].numel() != n:
                raise ValueError("Cycle count disagrees with transported records")
            result.append(row)
    return result
