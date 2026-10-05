"""Read-only, fixed-path diagnostics around a complete RL policy/head update.

KL and TV use the full vocabulary on sampled states, not the stored top-k teacher.
Acceptance chains on replayed paths are surrogates, distinct from rollout tau.
"""

from contextlib import contextmanager
import heapq
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F


def should_validate(step, total_steps, frequency, final_validation=False):
    return (frequency > 0 and (step % frequency == 0 or step >= total_steps)) or (
        final_validation and step >= total_steps
    )


def select_cases(records, max_cycles, max_context, seed):
    """Choose one uniformly sampled cycle per trajectory, then bound the candidate pool."""
    if max_cycles <= 0 or max_context <= 0:
        raise ValueError("Probe budgets must be positive")
    rng = random.Random(seed)
    candidates = []
    for record in records:
        eligible = []
        for cycle, position in enumerate(record["position"].tolist()):
            tokens = record["draft_tokens"][cycle].detach().cpu()
            depth = tokens.numel() - 1
            if depth < 1 or position < 0 or position >= record["prefix_ids"].numel():
                raise ValueError("Invalid probe cycle or incomplete committed prefix")
            if position + 1 + depth > max_context:
                continue
            if int(record["prefix_ids"][position]) != int(tokens[0]):
                raise ValueError("Probe seed does not match the committed prefix")
            eligible.append((cycle, position))
        if eligible:
            cycle, position = rng.choice(eligible)
            candidates.append((rng.random(), record, cycle, position))
    selected = heapq.nlargest(max_cycles, candidates, key=lambda item: item[0])
    return [{**cycle_case(record, cycle), "_sampling_priority": priority}
            for priority, record, cycle, _ in selected]


def cycle_case(record, cycle):
    """Build the same verifier path for diagnostics and full-trajectory selection."""
    position = int(record["position"][cycle])
    tokens = record["draft_tokens"][cycle].detach().cpu()
    actual = record["target_ids"][:position + 2].detach().cpu()
    shifted = record["prefix_ids"][:position + 1].detach().cpu()
    if (tokens.numel() < 2 or position < 0 or actual.numel() != position + 2
            or not torch.equal(actual[1:], shifted) or int(shifted[-1]) != int(tokens[0])):
        raise ValueError("Actual target context disagrees with the shifted MTP stream")
    return {"input_ids": torch.cat([actual[:-1], tokens[:-1]]), "draft_tokens": tokens,
            "position": position, "cycle_index": cycle,
            "trajectory_index": int(record["trajectory_index"]),
            "source_rank": int(record.get("source_rank", 0)),
            "advantage": float(record["advantage"]), "score": float(record["score"]),
            "verify_valid_length": int(record["accept_len"][cycle])}


def synchronized_cases(cases, max_cycles, group=None):
    """Every FSDP rank replays the same cases, including ranks with no local cycles."""
    if torch.distributed.is_initialized():
        parts = [None] * torch.distributed.get_world_size(group)
        torch.distributed.all_gather_object(parts, cases, group=group)
        cases = [case for part in parts for case in part]
    # Local top-K priorities contain every possible global top-K case. Unlike
    # sampling capped rank-local pools again, this weights eligible cycles equally.
    return heapq.nlargest(max_cycles, cases, key=lambda case: (
        case["_sampling_priority"], case["source_rank"], case["trajectory_index"], case["cycle_index"]
    ))


def compare_distributions(old_target, new_target, fixed_draft, updated_draft):
    """Compare normalized log probabilities [cycles, depth, vocabulary]."""
    tensors = [x.detach().float() for x in (old_target, new_target, fixed_draft, updated_draft)]
    if any(x.shape != tensors[0].shape or x.ndim != 3 for x in tensors):
        raise ValueError("Probe distributions must have matching [cycle, depth, vocab] shapes")
    if not tensors[0].numel() or any(not torch.isfinite(x).all() for x in tensors):
        raise ValueError("Probe distributions must be nonempty and finite")
    if any(not torch.allclose(x.logsumexp(-1), torch.zeros_like(x[..., 0]), atol=1e-4) for x in tensors):
        raise ValueError("Probe inputs must be normalized log probabilities")
    old, new, draft, updated = tensors
    p_old, p_new, q, q_updated = [x.exp() for x in tensors]
    kl = (p_new * (new - old)).sum(-1).clamp_min(0)
    tv = (p_new - p_old).abs().sum(-1) * 0.5
    alpha_pre = torch.minimum(p_old, q).sum(-1).clamp(0, 1)
    alpha_fixed = torch.minimum(p_new, q).sum(-1).clamp(0, 1)
    alpha_joint = torch.minimum(p_new, q_updated).sum(-1).clamp(0, 1)
    tau_pre, tau_fixed, tau_joint = [1 + x.cumprod(-1).sum(-1) for x in (alpha_pre, alpha_fixed, alpha_joint)]
    prefix = "draft/probe/"
    metrics = {
        prefix + "num_cycles": float(old.shape[0]),
        prefix + "num_positions": float(kl.numel()),
        prefix + "full_vocabulary": 1.0,
        prefix + "kl_new_old_mean": kl.mean().item(),
        prefix + "kl_new_old_max": kl.max().item(),
        prefix + "tv_new_old_mean": tv.mean().item(),
        prefix + "tau_pre_surrogate": tau_pre.mean().item(),
        prefix + "tau_post_fixed_draft_surrogate": tau_fixed.mean().item(),
        prefix + "delta_lag_surrogate": (tau_pre - tau_fixed).mean().item(),
        prefix + "tau_post_joint_surrogate": tau_joint.mean().item(),
        prefix + "joint_update_gain_surrogate": (tau_joint - tau_fixed).mean().item(),
    }
    for k in range(old.shape[1]):
        for name, values in (("alpha_pre", alpha_pre), ("alpha_post_fixed_draft", alpha_fixed),
                             ("alpha_post_joint", alpha_joint)):
            metrics[f"{prefix}{name}_{k + 1}"] = values[:, k].mean().item()
    rows = [{
        "kl_new_old_mean": kl[i].mean().item(),
        "tv_new_old_mean": tv[i].mean().item(),
        "tau_pre_surrogate": tau_pre[i].item(),
        "tau_post_fixed_draft_surrogate": tau_fixed[i].item(),
        "delta_lag_surrogate": (tau_pre[i] - tau_fixed[i]).item(),
        "tau_post_joint_surrogate": tau_joint[i].item(),
        "joint_update_gain_surrogate": (tau_joint[i] - tau_fixed[i]).item(),
        "alpha_pre": alpha_pre[i].tolist(),
        "alpha_post_fixed_draft": alpha_fixed[i].tolist(),
        "alpha_post_joint": alpha_joint[i].tolist(),
    } for i in range(old.shape[0])]
    return metrics, rows


@contextmanager
def probe_context(model):
    """Keep dropout/RNG, parameter shards and training flags unchanged by diagnostics."""
    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    sharded = hasattr(model, "unshard")
    head = model.mtp
    shard_settings = []
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            model.eval()
            if sharded:
                # A decoder-only call bypasses the root forward hook. Initialize the
                # root explicitly before any child can become an independent root.
                model._get_fsdp_state()._lazy_init()
                for module in (model, head):
                    state = module._get_fsdp_state()
                    group = state._fsdp_param_group
                    shard_settings.append((state, group, getattr(state, "_auto_reshard_after_forward", None),
                                           group.post_forward_mesh_info if group is not None else None))
                model.set_reshard_after_forward(False, recurse=False)
                model.unshard()
                head.set_reshard_after_forward(False, recurse=False)
                head.unshard()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                yield
    finally:
        if sharded:
            head.reshard()
            model.reshard()
            for state, group, automatic, mesh in shard_settings:
                if automatic is not None:
                    state._auto_reshard_after_forward = automatic
                if group is not None:
                    group.post_forward_mesh_info = mesh
        for module, mode in modes:
            module.training = mode


def _draft_logprobs(head, embed, lm_head, hidden, case):
    from transformers.cache_utils import DynamicCache

    anchor = case["position"]
    device = hidden.device
    ids = case["input_ids"].to(device)
    tokens = case["draft_tokens"].to(device)
    cache = DynamicCache()
    # Replay the committed prefix with target features, exactly as draft training does.
    for start in range(0, anchor, 256):
        end = min(start + 256, anchor)
        positions = torch.arange(start, end, device=device)[None]
        visible = torch.arange(end, device=device)[None] <= positions.T
        mask = hidden.new_zeros((1, 1, end - start, end)).masked_fill(~visible[None, None], -torch.inf)
        # MTP pairs target h(token_i) with embedding(token_{i+1}).
        head(hidden[:, start:end], F.embedding(ids[start + 1:end + 1], embed.weight)[None], positions, cache, mask)
    h = hidden[:, anchor:anchor + 1]
    output = []
    for depth in range(tokens.numel() - 1):
        position = torch.tensor([[anchor + depth]], device=device)
        mask = h.new_zeros((1, 1, 1, anchor + depth + 1))
        h = head(h, F.embedding(tokens[depth:depth + 1], embed.weight)[None], position, cache, mask)
        # SGLang selects drafts using raw head logits, irrespective of target temperature.
        logits = lm_head(h).float()
        output.append(logits.log_softmax(-1).squeeze(0).squeeze(0).cpu())
    return torch.stack(output)


def iter_evaluations(model, cases, temperature=1.0, *, include_draft=False, include_hidden=False):
    """Stream one cycle at a time, projecting only verifier positions."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Probe temperature must be positive and finite")
    core = getattr(model, "_fsdp_wrapped_module", model)
    if hasattr(core, "get_base_model"):
        core = core.get_base_model()  # PEFT adapters remain active inside the decoder.
    decoder = core.get_decoder() if hasattr(core, "get_decoder") else getattr(core.model, "language_model", core.model)
    embed = core.get_input_embeddings()
    with probe_context(model):
        for case in cases:
            ids = case["input_ids"].to(embed.weight.device)[None]
            positions = torch.arange(ids.shape[-1], device=ids.device)[None]
            hidden = decoder(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=positions,
                             use_cache=False, return_dict=True).last_hidden_state
            anchor = case["position"]
            logits = core.lm_head(hidden[:, anchor + 1:]).float() / temperature
            target = logits.log_softmax(-1).squeeze(0).cpu()
            draft = _draft_logprobs(model.mtp, embed, core.lm_head, hidden, case) if include_draft else None
            features = hidden.squeeze(0).cpu() if include_hidden else None
            yield target, draft, features


def evaluate_cases(model, cases, temperature=1.0, return_hidden=False):
    """Bounded diagnostic API; training uses the streaming target-only evaluator."""
    if not cases:
        raise ValueError("No probe cases to evaluate")
    target_rows, draft_rows, hidden_rows = [], [], []
    for target, draft, hidden in iter_evaluations(model, cases, temperature,
                                                include_draft=True, include_hidden=return_hidden):
        target_rows.append(target)
        draft_rows.append(draft)
        if return_hidden:
            hidden_rows.append(hidden)
    distributions = torch.stack(target_rows), torch.stack(draft_rows)
    return (*distributions, hidden_rows) if return_hidden else distributions


def write_probe_rows(output_dir, step, cases, rows):
    """One file per diagnostic step; omit token/hidden tensors from the scalar report."""
    if not output_dir:
        return
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{step}.jsonl").open("w", encoding="utf-8") as stream:
        for case, row in zip(cases, rows, strict=True):
            metadata = {key: value for key, value in case.items()
                        if not isinstance(value, torch.Tensor) and not key.startswith("_")}
            priority = row.get("positive_advantage_kl_score",
                               max(float(case["advantage"]), 0.0) * row["kl_new_old_mean"])
            stream.write(json.dumps({"step": step, **metadata, **row,
                                     "positive_advantage_kl_score": priority if math.isfinite(priority) else None},
                                    allow_nan=False) + "\n")


def write_trajectory_rows(output_dir, step, rows):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{step}.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps({"step": step, **row}, allow_nan=False) + "\n")
