"""FSDP2 integration: one isolated head group, bounded backward, one gradient flush."""

import math
import torch
import torch.distributed as dist
from transformers.cache_utils import DynamicCache
from verl.models.transformers.mtp import shared_layers
from .replay import replay_chunks, attention_backend
from .signals import unpack_signals
from .loss import UNCOMPUTED_ALPHA


def optimizer_groups(model, config):
    head_ids = {id(p) for p in model.mtp.parameters()}
    return [
        {
            "params": [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad],
            "name": "policy",
        },
        {"params": list(model.mtp.parameters()), "lr": config.learning_rate, "name": "mtp"},
    ]


def head_schedule(step, total_steps, warmup_steps, min_ratio):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def backward_head(model, data, config, global_cycles, dp_size):
    """Accumulate the global cycle mean after policy backward, before the shared optimizer step.

    FSDP averages reduced gradients, hence local sums are scaled by dp_size/global_cycles.
    All ranks issue one head reduce-scatter even if cycle/chunk counts differ.
    """
    if "mtp_num_cycles" not in data:
        raise RuntimeError("GrowMTP update received no verification-record transport fields")
    if global_cycles == 0:
        return {"mtp/num_cycles": 0.0, "mtp/dca_loss": 0.0}
    head = model.mtp
    embed, lm = shared_layers(model)
    fsdp = hasattr(head, "unshard")
    records = unpack_signals(data)
    if fsdp:
        model.set_reshard_after_forward(False, recurse=False)
        model.set_reshard_after_backward(False, recurse=False)
        model.unshard()
        head.reshard()
        head.unshard()
        head.set_reshard_after_forward(False)
        head.set_reshard_after_backward(False)
        head.set_requires_gradient_sync(False)
    device = embed.weight.device
    total, count = torch.zeros((), device=device), 0
    alpha_sum = torch.zeros(config.speculative_num_steps, device=device)
    alpha_count = torch.zeros(config.speculative_num_steps, device=device, dtype=torch.long)
    try:
        with torch.autocast(
            device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and embed.weight.dtype == torch.bfloat16,
        ):
            for record in records:
                if record["topk_val"].shape[1] != config.speculative_num_steps:
                    raise ValueError("Rollout and training draft depths disagree")
                for losses, alpha in replay_chunks(
                    head,
                    embed.weight,
                    lm.weight,
                    record,
                    config.chunk_size,
                    config.vocab_chunk_size,
                ):
                    (
                        losses.sum() * config.mtp_loss_scaling_factor * dp_size / global_cycles
                    ).backward()
                    total += losses.detach().sum()
                    count += losses.numel()
                    computed = alpha != UNCOMPUTED_ALPHA
                    alpha_sum += alpha.masked_fill(~computed, 0).sum(0)
                    alpha_count += computed.sum(0)
            if not records:
                # Run the same head graph with zero weight, so the flush has every parameter
                # even on a rank whose micro-batch contains only immediately finished requests.
                h = torch.zeros(
                    1, 1, head.config.hidden_size, device=device, dtype=embed.weight.dtype
                )
                pos = torch.zeros(1, 1, device=device, dtype=torch.long)
                with attention_backend(device):
                    out = head(
                        h,
                        h,
                        pos,
                        DynamicCache(),
                        torch.zeros(1, 1, 1, 1, device=device, dtype=h.dtype),
                    )
                (out.sum() * 0).backward()
        if fsdp:
            head.set_requires_gradient_sync(True)
            head._get_fsdp_state()._fsdp_param_group.post_backward()
    finally:
        if fsdp:
            head.set_requires_gradient_sync(True)
            head.reshard()
            model.reshard()
            model.set_reshard_after_backward(True, recurse=False)
    metrics = {"mtp/num_cycles": float(count), "mtp/dca_loss": total.item() / max(1, count)}
    counts = alpha_count.tolist()
    metrics["mtp/alpha_valid_only"] = 1.0
    metrics["mtp/projected_steps"] = sum(counts)
    metrics["mtp/projected_steps_fraction"] = sum(counts) / max(1, count * config.speculative_num_steps)
    for i, (value, valid_count) in enumerate(zip(alpha_sum.tolist(), counts)):
        metrics[f"mtp/alpha_{i+1}_valid_count"] = valid_count
        if valid_count:
            metrics[f"mtp/alpha_{i+1}"] = value / valid_count
    return metrics
