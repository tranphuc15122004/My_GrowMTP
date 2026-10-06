"""Focused checks for rollout-token alignment and auxiliary CE gradients."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def load_mtp_file(name):
    path = ROOT / "verl/verl/trainer/mtp" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"growmtp_{name}_aux_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rollout_labels_include_rejection_correction_and_exclude_truncated_tokens():
    signals = load_mtp_file("signals")
    signal = {
        "prompt_ids": torch.tensor([2, 3, 10]),
        "position": [2, 4],
        "accept_len": [2, 4],
        "topk_idx": [torch.zeros(3, 2, dtype=torch.long)] * 2,
        "accepted_tokens": [torch.tensor([11, 12]), torch.tensor([13, 14, 15, 16])],
    }
    # Token 10 is the target seed. Cycle 0 accepts draft 11, then emits
    # correction 12. Cycle 1 accepts all drafts; bonus 16 has no draft logit.
    labels, mask = signals._rollout_targets(
        signal,
        torch.tensor([10, 11, 12, 13, 14, 15, 16, 0]),
        torch.tensor([1, 1, 1, 1, 1, 0, 0, 0]),
    )
    assert labels.tolist() == [[11, 12, 0], [13, 14, 0]]
    assert mask.tolist() == [[True, True, False], [True, True, False]]
    with pytest.raises(ValueError, match="verification history"):
        signals._rollout_targets(
            signal,
            torch.tensor([10, 99, 12, 13, 14, 15, 16, 0]),
            torch.tensor([1, 1, 1, 1, 1, 0, 0, 0]),
        )


def test_target_labels_survive_ragged_signal_transport():
    signals = load_mtp_file("signals")
    prompt_hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    signal = {
        "prompt_ids": torch.tensor([2, 3, 10]),
        "prompt_hidden": prompt_hidden,
        "position": [2],
        "accept_len": [2],
        "hidden": [prompt_hidden[-1]],
        "draft_tokens": [torch.tensor([10, 11, 99, 98])],
        "topk_val": [torch.zeros(3, 2)],
        "topk_idx": [torch.zeros(3, 2, dtype=torch.long)],
        "accepted_hidden": [torch.zeros(2, 4)],
        "accepted_tokens": [torch.tensor([11, 12])],
    }
    batch = {
        "prompts": torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]]),
        "responses": torch.tensor([[10, 11, 12, 0], [0, 0, 0, 0]]),
        "response_mask": torch.tensor([[1, 1, 1, 0], [0, 0, 0, 0]]),
    }
    signals.pack_signals(batch, [signal, None], include_target_tokens=True)
    records = signals.unpack_signals(batch)
    assert len(records) == 1
    assert records[0]["batch_index"] == 0
    assert records[0]["target_tokens"].tolist() == [[11, 12, 0]]
    assert records[0]["target_mask"].tolist() == [[True, True, False]]

    bad_batch = {key: batch[key].clone() for key in
                 ("prompts", "attention_mask", "responses", "response_mask")}
    bad_batch["prompts"][0, 2] = 9
    with pytest.raises(ValueError, match="shifted prompt"):
        signals.pack_signals(bad_batch, [signal, None], include_target_tokens=True)


@pytest.mark.parametrize("lengths", ([1, 2, 4], [4, 4, 4]))
def test_aux_ce_matches_full_vocab_reference_and_keeps_dca_gradient(lengths):
    loss_module = load_mtp_file("loss")
    torch.manual_seed(7)
    hidden = torch.randn(3, 3, 5, requires_grad=True)
    weight = torch.randn(13, 5)
    teacher_logits = torch.randn(3, 3, 13)
    teacher_logprobs, teacher_ids = teacher_logits.log_softmax(-1).topk(4, dim=-1)
    valid_lengths = torch.tensor(lengths)
    target_ids = torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
    target_mask = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 0, 0]], dtype=torch.bool)

    base_dca, base_alpha = loss_module.chunked_dca(
        hidden, weight, teacher_logprobs, teacher_ids, valid_lengths, chunk_size=1
    )
    dca, alpha, ce = loss_module.chunked_dca(
        hidden, weight, teacher_logprobs, teacher_ids, valid_lengths,
        chunk_size=1, target_ids=target_ids, target_mask=target_mask,
    )
    logits = F.linear(hidden, weight)
    reference_dca, _ = loss_module.dca_loss(logits, teacher_logprobs, teacher_ids, valid_lengths)
    reachable = torch.arange(3)[None, :] < valid_lengths[:, None]
    valid = target_mask & reachable
    token_ce = -logits.log_softmax(-1).gather(-1, target_ids[..., None]).squeeze(-1)
    reference_ce = (token_ce * valid).sum(-1) / valid.sum(-1).clamp_min(1)

    torch.testing.assert_close(dca, base_dca)
    torch.testing.assert_close(dca, reference_dca)
    torch.testing.assert_close(alpha, base_alpha)
    torch.testing.assert_close(ce, reference_ce)
    assert ce[-1].item() == 0

    aux_gradient = torch.autograd.grad((dca + 0.2 * ce).sum(), hidden)[0]
    reference_gradient = torch.autograd.grad((reference_dca + 0.2 * reference_ce).sum(), hidden)[0]
    torch.testing.assert_close(aux_gradient, reference_gradient)
