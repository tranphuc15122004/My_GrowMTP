"""Classic single-chain rejection sampling with the actual draft distribution."""

import torch


def classic_rejection_sample(target_probs, draft_tokens, draft_probs):
    """Return accepted-prefix lengths and replacement/bonus tokens.

    target_probs: [B, K+1, V], draft_tokens: [B, K], draft_probs: [B, K, V].
    The proposal tokens must have been sampled from draft_probs. Target probabilities
    may already include temperature/top-p processing; no approximation is used here.
    """
    b, k = draft_tokens.shape
    if target_probs.shape[1] != k + 1 or draft_probs.shape != target_probs[:, :k].shape:
        raise ValueError("Target and draft distribution shapes disagree")
    p = target_probs[:, :k].gather(-1, draft_tokens[..., None]).squeeze(-1)
    q = draft_probs.gather(-1, draft_tokens[..., None]).squeeze(-1)
    ratio = (p / q.clamp_min(torch.finfo(q.dtype).tiny)).clamp(max=1)
    accepted = (torch.rand_like(ratio) < ratio).to(torch.int32).cumprod(-1)
    lengths = accepted.sum(-1).to(torch.int32)
    rows = torch.arange(b, device=target_probs.device)
    distribution = target_probs[rows, lengths.long()]
    q_rejected = draft_probs[rows, lengths.long().clamp(max=k - 1)]
    residual = (distribution - q_rejected).clamp_min(0)
    mass = residual.sum(-1, keepdim=True)
    # At a true rejection the residual has positive mass. A zero mass can occur
    # from finite-precision cancellation; the target distribution is then the limit.
    residual = torch.where(
        mass > 0, residual / mass.clamp_min(torch.finfo(mass.dtype).tiny), distribution
    )
    distribution = torch.where((lengths < k)[:, None], residual, distribution)
    bonus = torch.multinomial(distribution, 1).squeeze(-1)
    return lengths, bonus
