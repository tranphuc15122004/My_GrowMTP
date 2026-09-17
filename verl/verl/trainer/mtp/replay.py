"""Draft-path reconstruction against the full committed-prefix KV (Appendix E)."""

from contextlib import nullcontext
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from .loss import chunked_dca


def attention_backend(device):
    if device.type != "cuda":
        return nullcontext()
    from torch.nn.attention import sdpa_kernel, SDPBackend

    # Arbitrary masks require efficient attention. Avoid per-shape cuDNN engine growth.
    return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])


def clone_prefix(cache):
    result = DynamicCache()
    for i, layer in enumerate(cache.layers):
        result.update(layer.keys.detach(), layer.values.detach(), i)
    return result


def replay_mask(anchors, prefix_length, depth, dtype):
    """Causal committed-prefix visibility; diagonal visibility at every draft depth."""
    c = anchors.numel()
    prefix = torch.arange(prefix_length, device=anchors.device)[None, :] < anchors[:, None]
    draft = torch.eye(c, device=anchors.device, dtype=torch.bool).repeat(1, depth + 1)
    visible = torch.cat([prefix, draft], dim=-1)
    return torch.zeros(1, 1, c, visible.shape[-1], device=anchors.device, dtype=dtype).masked_fill(
        ~visible[None, None], -torch.inf
    )


def prefill_prefix(head, embed_weight, record, stop, chunk_size=256):
    """Rebuild keys/values from every committed token's recorded target feature."""
    cache = DynamicCache()
    device = embed_weight.device
    h = record["prefix_hidden"][:stop].to(device=device, dtype=embed_weight.dtype)
    ids = record["prefix_ids"][:stop].to(device)
    with torch.no_grad(), attention_backend(device):
        for start in range(0, stop, chunk_size):
            end = min(start + chunk_size, stop)
            positions = torch.arange(start, end, device=device)[None]
            visible = torch.arange(end, device=device)[None, :] <= positions.T
            mask = torch.zeros(1, 1, end - start, end, device=device, dtype=h.dtype)
            mask.masked_fill_(~visible[None, None], -torch.inf)
            head(
                h[start:end][None],
                F.embedding(ids[start:end], embed_weight)[None],
                positions,
                cache,
                mask,
            )
    return clone_prefix(cache)


def replay_chunks(head, embed_weight, lm_weight, record, chunk_size=1024, vocab_chunk_size=256):
    """Yield disjoint chunk graphs; only recorded prefix constants cross chunks."""
    device = embed_weight.device
    embed_weight, lm_weight = embed_weight.detach(), lm_weight.detach()
    positions = record["position"].detach().cpu()
    n = positions.numel()
    if not n:
        return
    if (positions < 0).any() or (positions[1:] <= positions[:-1]).any():
        raise ValueError("Cycle starts must be nonnegative and strictly increasing")
    stop = int(positions[-1])
    anchors = positions.to(device)
    if record["prefix_hidden"].shape[0] <= stop or record["prefix_ids"].shape[0] <= stop:
        raise ValueError("Committed prefix is incomplete for the final cycle")
    # The prompt already contains the first seed. Exclude each seed from prefix visibility;
    # replay it once at its original position, not at position+1.
    prefix = prefill_prefix(head, embed_weight, record, stop)
    k = record["topk_val"].shape[1]
    if record["draft_tokens"].shape[1] != k + 1:
        raise ValueError("A cycle must carry its seed followed by K draft tokens")
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        pos = anchors[start:end]
        tokens = record["draft_tokens"][start:end].to(device)
        h = record["hidden"][start:end].to(device=device, dtype=embed_weight.dtype).detach()[None]
        cache = clone_prefix(prefix)
        outputs = []
        with attention_backend(device):
            for depth in range(k):
                emb = F.embedding(tokens[:, depth], embed_weight)[None]
                h = head(h, emb, (pos + depth)[None], cache, replay_mask(pos, stop, depth, h.dtype))
                outputs.append(h.squeeze(0))
        hidden = torch.stack(outputs, dim=1)
        loss, alpha = chunked_dca(
            hidden,
            lm_weight,
            record["topk_val"][start:end].to(device),
            record["topk_idx"][start:end].to(device),
            record["accept_len"][start:end],
            vocab_chunk_size,
        )
        yield loss, alpha
