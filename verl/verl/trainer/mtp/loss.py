"""Depth-Coupled Acceptance (DCA) loss with Verify-Gated Masking (VGM)."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

UNCOMPUTED_ALPHA = -1.0


def _validate_lengths(valid_lengths, shape, k):
    if valid_lengths.shape != shape:
        raise ValueError("One valid length is required per cycle")
    if ((valid_lengths < 1) | (valid_lengths > k + 1)).any():
        raise ValueError("Verify lengths must be in [1, K+1]; missing lengths are invalid")


def acceptance_overlap(logits, teacher_logprobs, teacher_ids):
    """Top-k plus one residual bin. Teacher probabilities keep full-vocabulary normalization."""
    p = teacher_logprobs.detach().float().exp()
    q = F.softmax(logits.float(), dim=-1).gather(-1, teacher_ids)
    return (
        torch.minimum(p, q).sum(-1)
        + torch.minimum((1 - p.sum(-1)).clamp_min(0), (1 - q.sum(-1)).clamp_min(0))
    ).clamp(0, 1)


def acceptance_overlap_and_ce(logits, teacher_logprobs, teacher_ids, target_ids):
    """Compute DCA overlap and rollout-token CE from one full-vocabulary normalizer."""
    logits = logits.float()
    log_z = logits.logsumexp(-1)
    p = teacher_logprobs.detach().float().exp()
    q_topk = (logits.gather(-1, teacher_ids) - log_z.unsqueeze(-1)).exp()
    alpha = (
        torch.minimum(p, q_topk).sum(-1)
        + torch.minimum((1 - p.sum(-1)).clamp_min(0), (1 - q_topk.sum(-1)).clamp_min(0))
    ).clamp(0, 1)
    target_logprob = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1) - log_z
    return alpha, -target_logprob


def dca_from_alpha(alpha, valid_lengths):
    """Per-cycle loss; lengths include the first rejected draft (or K for full acceptance)."""
    k = alpha.shape[-1]
    _validate_lengths(valid_lengths, alpha.shape[:-1], k)
    reachable = torch.arange(k, device=alpha.device) < valid_lengths.to(alpha.device)[..., None]
    chain = alpha.clamp_min(1e-6).log().cumsum(-1)
    return -torch.logsumexp(chain.masked_fill(~reachable, -torch.inf), dim=-1)


def dca_loss(logits, teacher_logprobs, teacher_ids, valid_lengths):
    """Per-cycle DCA with VGM, as defined in paper Eq. 9."""
    alpha = acceptance_overlap(logits, teacher_logprobs, teacher_ids)
    return dca_from_alpha(alpha, valid_lengths), alpha.detach()


def chunked_dca(
    hidden,
    weight,
    teacher_logprobs,
    teacher_ids,
    valid_lengths,
    chunk_size=256,
    *,
    target_ids=None,
    target_mask=None,
):
    """Project VGM-reachable positions with bounded storage and checkpoint recomputation.

    Returned alpha entries outside the supervised prefix are UNCOMPUTED_ALPHA.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    n, k, hidden_size = hidden.shape
    _validate_lengths(valid_lengths, hidden.shape[:-2], k)
    if (target_ids is None) != (target_mask is None):
        raise ValueError("Rollout target IDs and mask must be provided together")
    if target_ids is not None and (target_ids.shape != (n, k) or target_mask.shape != (n, k)):
        raise ValueError("Rollout target IDs and mask must have one value per draft position")
    weight = weight.detach()
    reachable = torch.arange(k, device=hidden.device) < valid_lengths.to(hidden.device)[:, None]
    indices = reachable.reshape(-1).nonzero(as_tuple=False).flatten()
    if indices.numel() != n * k:
        h = hidden.reshape(-1, hidden_size).index_select(0, indices)
        topk = teacher_ids.shape[-1]
        p = teacher_logprobs.reshape(-1, topk).index_select(0, indices)
        ids = teacher_ids.reshape(-1, topk).index_select(0, indices)
        active_target_ids = (
            target_ids.reshape(-1).index_select(0, indices).to(hidden.device)
            if target_ids is not None else None
        )

        def project(h, p, ids, labels=None):
            logits = F.linear(h, weight)
            if labels is None:
                return acceptance_overlap(logits, p, ids)
            return acceptance_overlap_and_ce(logits, p, ids, labels)

        alpha_parts, ce_parts = [], []
        row_limit = chunk_size * k
        for start in range(0, h.shape[0], row_limit):
            stop = start + row_limit
            if target_ids is None:
                alpha_parts.append(checkpoint(
                    project, h[start:stop], p[start:stop], ids[start:stop], use_reentrant=False
                ))
            else:
                labels = active_target_ids[start:stop]
                active_alpha, active_ce = checkpoint(
                    project, h[start:stop], p[start:stop], ids[start:stop], labels,
                    use_reentrant=False,
                )
                alpha_parts.append(active_alpha)
                ce_parts.append(active_ce)
        active_alpha = torch.cat(alpha_parts)
        alpha = active_alpha.new_full((n * k,), UNCOMPUTED_ALPHA)
        alpha = alpha.index_copy(0, indices, active_alpha).reshape(n, k)
        lengths = reachable.sum(-1)
        losses = active_alpha.new_zeros(n)
        for length in range(1, k + 1):
            rows = (lengths == length).nonzero(as_tuple=False).flatten()
            if rows.numel():
                prefix = alpha.index_select(0, rows)[:, :length]
                chain = prefix.clamp_min(1e-6).log().cumsum(-1)
                losses = losses.index_copy(0, rows, -torch.logsumexp(chain, dim=-1))
        if target_ids is None:
            return losses, alpha.detach()
        active_ce = torch.cat(ce_parts)
        ce_values = active_ce.new_zeros(n * k).index_copy(0, indices, active_ce).reshape(n, k)
        ce_mask = target_mask.to(device=hidden.device, dtype=torch.bool) & reachable
        ce_count = ce_mask.sum(-1)
        ce_losses = (ce_values * ce_mask).sum(-1) / ce_count.clamp_min(1)
        return losses, alpha.detach(), ce_losses

    def compute(h, p, ids, lengths, labels=None, labels_mask=None):
        logits = F.linear(h, weight)
        if labels is None:
            return dca_loss(logits, p, ids, lengths)
        alpha, ce_values = acceptance_overlap_and_ce(logits, p, ids, labels)
        losses = dca_from_alpha(alpha, lengths)
        chunk_reachable = torch.arange(k, device=hidden.device) < lengths.to(hidden.device)[:, None]
        ce_mask = labels_mask.to(device=hidden.device, dtype=torch.bool) & chunk_reachable
        ce_count = ce_mask.sum(-1)
        ce_losses = (ce_values * ce_mask).sum(-1) / ce_count.clamp_min(1)
        return losses, alpha.detach(), ce_losses

    losses, alphas, ce_losses = [], [], []
    for start in range(0, hidden.shape[0], chunk_size):
        stop = start + chunk_size
        if target_ids is None:
            loss, alpha = checkpoint(
                compute,
                hidden[start:stop],
                teacher_logprobs[start:stop],
                teacher_ids[start:stop],
                valid_lengths[start:stop],
                use_reentrant=False,
            )
        else:
            loss, alpha, ce_loss = checkpoint(
                compute,
                hidden[start:stop],
                teacher_logprobs[start:stop],
                teacher_ids[start:stop],
                valid_lengths[start:stop],
                target_ids[start:stop].to(hidden.device),
                target_mask[start:stop].to(hidden.device),
                use_reentrant=False,
            )
            ce_losses.append(ce_loss)
        losses.append(loss)
        alphas.append(alpha)
    result = (torch.cat(losses), torch.cat(alphas))
    if target_ids is None:
        return result
    return (*result, torch.cat(ce_losses))
