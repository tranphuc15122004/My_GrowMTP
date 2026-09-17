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


def pack_signals(batch, records):
    """Remove Python-object payloads at the rollout boundary; keep one nested tensor per field."""
    present = [s for s in records if s and len(s.get("position", []))]
    if not present:
        batch["mtp_num_cycles"] = torch.zeros(len(records), dtype=torch.int64)
        return
    prefixes = {id(signal): _prefix_inputs(signal) for signal in present}
    rows = {key: [] for key in FIELDS}
    counts = []
    prototype = present[0]
    for signal in records:
        n = len(signal["position"]) if signal else 0
        counts.append(n)
        source = signal if n else prototype
        for key in FIELDS:
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
    rows = {key: data["mtp_" + key].unbind() for key in FIELDS
            if key not in ("position", "accept_len")}
    for key in ("position", "accept_len"):
        rows[key] = data["mtp_" + key].detach().cpu().unbind()
    result = []
    for i, n in enumerate(counts):
        if n:
            row = {key: rows[key][i] for key in FIELDS}
            if row["position"].numel() != n or row["accept_len"].numel() != n:
                raise ValueError("Cycle count disagrees with transported records")
            result.append(row)
    return result
