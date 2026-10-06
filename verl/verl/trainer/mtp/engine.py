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


def backward_head(model, data, config, global_cycles, dp_size, *, global_trajectories=None):
    """Accumulate the global cycle mean after policy backward, before the shared optimizer step.

    FSDP averages reduced gradients, hence local sums are scaled by dp_size/global_cycles.
    All ranks issue one head reduce-scatter even if cycle/chunk counts differ.
    Deferred refresh steps optionally use a trajectory mean: average cycles within
    each trajectory, then average over the full rollout batch (including empty rows).
    """
    if "mtp_num_cycles" not in data:
        raise RuntimeError("GrowMTP update received no verification-record transport fields")
    if global_cycles == 0:
        return {"mtp/num_cycles": 0.0, "mtp/dca_loss": 0.0}
    head = model.mtp
    embed, lm = shared_layers(model)
    fsdp = hasattr(head, "unshard")
    records = unpack_signals(data)
    aux_lambda = float(getattr(config, "rollout_aux_ce_lambda", 0.0))
    advantage_clip = float(getattr(config, "rollout_aux_advantage_clip", 2.0))
    if not math.isfinite(aux_lambda) or aux_lambda < 0:
        raise ValueError("rollout_aux_ce_lambda must be finite and nonnegative")
    if not math.isfinite(advantage_clip) or advantage_clip <= 0:
        raise ValueError("rollout_aux_advantage_clip must be finite and positive")
    if aux_lambda > 0:
        if global_trajectories is not None:
            raise ValueError("Rollout auxiliary CE must be piloted separately from exact-KL refresh")
        if "advantages" not in data or "response_mask" not in data:
            raise RuntimeError("Auxiliary CE requires rollout advantages and response masks")
        from .metrics import trajectory_advantages

        trajectory_values = trajectory_advantages(data["advantages"], data["response_mask"])
        trajectory_values = trajectory_values.detach().float().cpu()
        for record in records:
            value = float(trajectory_values[record["batch_index"]])
            record["aux_advantage"] = min(max(value, 0.0), advantage_clip) if math.isfinite(value) else 0.0
            if "target_tokens" not in record or "target_mask" not in record:
                raise RuntimeError("Auxiliary CE is enabled but rollout target labels were not packed")
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
    total = torch.zeros((), device=device)
    aux_total = torch.zeros((), device=device)
    aux_unweighted_total = torch.zeros((), device=device)
    count, aux_valid_cycles, aux_positive_cycles = 0, 0, 0
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
                denominator = global_cycles
                if global_trajectories is not None:
                    denominator = global_trajectories * record["position"].numel()
                if aux_lambda > 0:
                    valid_target_cycles = record["target_mask"].any(-1)
                    aux_valid_cycles += int(valid_target_cycles.sum())
                    if record["aux_advantage"] > 0:
                        aux_positive_cycles += int(valid_target_cycles.sum())
                record_aux_ce = aux_lambda > 0 and record["aux_advantage"] > 0 and bool(
                    record["target_mask"].any()
                )
                replay_options = {"include_aux_ce": True} if record_aux_ce else {}
                for result in replay_chunks(
                    head,
                    embed.weight,
                    lm.weight,
                    record,
                    config.chunk_size,
                    config.vocab_chunk_size,
                    **replay_options,
                ):
                    if record_aux_ce:
                        losses, alpha, aux_ce = result
                        weight = record["aux_advantage"]
                        weighted_aux = aux_ce * weight
                        combined_loss = losses + aux_lambda * weighted_aux
                        aux_total += weighted_aux.detach().sum()
                        aux_unweighted_total += aux_ce.detach().sum()
                    else:
                        losses, alpha = result
                        combined_loss = losses
                    (
                        combined_loss.sum()
                        * config.mtp_loss_scaling_factor
                        * dp_size
                        / denominator
                    ).backward()
                    total += losses.detach().sum() / (record["position"].numel()
                                                     if global_trajectories is not None else 1)
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
    mean_count = len(data["mtp_num_cycles"]) if global_trajectories is not None else count
    metrics = {"mtp/num_cycles": float(count), "mtp/dca_loss": total.item() / max(1, mean_count)}
    if aux_lambda > 0:
        metrics.update({
            "mtp/aux_ce_lambda": aux_lambda,
            "mtp/aux_ce_advantage_clip": advantage_clip,
            "mtp/aux_ce_loss": aux_total.item() / max(1, mean_count),
            "mtp/aux_ce_unweighted_loss": aux_unweighted_total.item() / max(1, mean_count),
            "mtp/aux_ce_valid_cycles": float(aux_valid_cycles),
            "mtp/aux_ce_positive_cycles": float(aux_positive_cycles),
            "mtp/aux_ce_contribution": aux_lambda * aux_total.item() / max(1, mean_count),
        })
    counts = alpha_count.tolist()
    metrics["mtp/alpha_valid_only"] = 1.0
    metrics["mtp/projected_steps"] = sum(counts)
    metrics["mtp/projected_steps_fraction"] = sum(counts) / max(1, count * config.speculative_num_steps)
    for i, (value, valid_count) in enumerate(zip(alpha_sum.tolist(), counts)):
        metrics[f"mtp/alpha_{i+1}_valid_count"] = valid_count
        if valid_count:
            metrics[f"mtp/alpha_{i+1}"] = value / valid_count
    return metrics
