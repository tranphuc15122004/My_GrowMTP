"""Small comparison helpers that can be used without importing Ray or the trainer."""

import math

import torch


def trajectory_advantages(advantages: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Return one masked token mean per trajectory, with NaN for empty masks.

    Padded tensors have shape ``[trajectories, tokens]``. Nested tensors may
    contain different token lengths. Padding is selected out before reduction,
    so even nonfinite values in padding cannot affect a trajectory's mean.
    """
    if not advantages.is_nested and advantages.ndim != 2:
        raise ValueError("advantages must have shape [trajectories, tokens]")
    if not response_mask.is_nested and response_mask.ndim != 2:
        raise ValueError("response_mask must have shape [trajectories, tokens]")
    if not advantages.is_nested and not response_mask.is_nested:
        if advantages.shape != response_mask.shape:
            raise ValueError("advantages and response_mask must have the same shape")
        values = advantages.float() if not advantages.is_floating_point() else advantages
        mask = response_mask.to(device=values.device, dtype=torch.bool)
        return values.masked_fill(~mask, 0).sum(-1) / mask.sum(-1)
    advantage_rows = advantages.unbind(0)
    mask_rows = response_mask.unbind(0)
    if len(advantage_rows) != len(mask_rows):
        raise ValueError("advantages and response_mask have different trajectory counts")
    means = []
    for values, mask in zip(advantage_rows, mask_rows, strict=True):
        if values.shape != mask.shape:
            raise ValueError("advantages and response_mask have different trajectory shapes")
        values = values.float() if not values.is_floating_point() else values
        selected = values.masked_select(mask.to(device=values.device, dtype=torch.bool))
        means.append(selected.mean() if selected.numel() else values.new_full((), float("nan")))
    if means:
        return torch.stack(means)
    return torch.empty(0, dtype=torch.float32, device=advantages.device)


def trajectory_statistics(advantages: torch.Tensor, response_mask: torch.Tensor) -> dict[str, float]:
    """Summarize trajectory means equally, excluding empty or nonfinite means.

    Population standard deviation is used. Empty/nonfinite inputs produce NaN
    statistics, which the scalar JSONL logger stores as null, rather than zero.
    Counts distinguish empty masks from nonfinite advantages on valid tokens.
    """
    means = trajectory_advantages(advantages, response_mask)
    valid = means[torch.isfinite(means)]
    if response_mask.is_nested:
        empty_count = sum(int(not mask.bool().any().item()) for mask in response_mask.unbind(0))
    else:
        empty_count = response_mask.bool().sum(-1).eq(0).sum().item()
    prefix = "comparison/advantage/"
    result = {
        prefix + "total_trajectories": means.numel(),
        prefix + "valid_trajectories": valid.numel(),
        prefix + "empty_trajectories": empty_count,
        prefix + "nonfinite_trajectories": means.numel() - valid.numel() - empty_count,
    }
    if valid.numel():
        result.update({
            prefix + "mean": valid.mean().item(),
            prefix + "std": valid.std(unbiased=False).item(),
            prefix + "positive_fraction": (valid > 0).float().mean().item(),
            prefix + "negative_fraction": (valid < 0).float().mean().item(),
            prefix + "zero_fraction": (valid == 0).float().mean().item(),
        })
    else:
        result.update({prefix + name: float("nan") for name in
                       ("mean", "std", "positive_fraction", "negative_fraction", "zero_fraction")})
    return result


class ComparisonTracker:
    """Accumulate measured time and generated tokens for this process session.

    ``timing_raw['step']`` encloses training and diagnostic probing, and excludes
    ``testing``. One optional probe timer named ``mtp_probe`` or ``probe`` is
    subtracted from training time. Observed E2E is step plus testing, including
    probing and checkpointing inside step. Startup and unmeasured outside work
    are not estimated. Testing-only updates can account for initial validation.

    Session counters intentionally restart on checkpoint resume. Offline
    comparisons must sum deduplicated per-step metrics, not these counters.
    """

    def __init__(self):
        self.training_seconds = 0.0
        self.evaluation_seconds = 0.0
        self.probe_seconds = 0.0
        self.e2e_seconds = 0.0
        self.generated_tokens = 0
        self.gpu_hours = 0.0

    def update(self, timing_raw: dict, n_gpus: int, response_tokens: int) -> dict[str, float]:
        """Return scalar per-step and session metrics; reject invalid timing."""
        if "probe" in timing_raw and "mtp_probe" in timing_raw:
            raise ValueError("Use only one probe timer: probe or mtp_probe")
        step = self._seconds(timing_raw, "step")
        evaluation = self._seconds(timing_raw, "testing")
        probe_key = "mtp_probe" if "mtp_probe" in timing_raw else "probe"
        probe = self._seconds(timing_raw, probe_key)
        if probe > step:
            raise ValueError("Probe time cannot exceed the enclosing step time")
        if not math.isfinite(float(n_gpus)) or n_gpus <= 0:
            raise ValueError("n_gpus must be positive and finite")
        if not math.isfinite(float(response_tokens)) or response_tokens < 0 or int(response_tokens) != response_tokens:
            raise ValueError("response_tokens must be a nonnegative integer")
        training = step - probe
        e2e = step + evaluation
        gpu_hours = e2e * n_gpus / 3600.0
        self.training_seconds += training
        self.evaluation_seconds += evaluation
        self.probe_seconds += probe
        self.e2e_seconds += e2e
        self.generated_tokens += int(response_tokens)
        self.gpu_hours += gpu_hours
        result = {
            "comparison/schema_version": 1,
            "comparison/config/n_gpus": n_gpus,
            "comparison/time/step_evaluation_s": evaluation,
            "comparison/time/step_probe_s": probe,
            "comparison/time/step_e2e_s": e2e,
            "comparison/time/step_gpu_hours": gpu_hours,
            "comparison/time/session_training_s": self.training_seconds,
            "comparison/time/session_evaluation_s": self.evaluation_seconds,
            "comparison/time/session_probe_s": self.probe_seconds,
            "comparison/time/session_e2e_s": self.e2e_seconds,
            "comparison/time/session_gpu_hours": self.gpu_hours,
            "comparison/tokens/step_generated": int(response_tokens),
            "comparison/tokens/session_generated": self.generated_tokens,
        }
        if "step" in timing_raw:
            result["comparison/time/step_training_s"] = training
        return result

    @staticmethod
    def _seconds(timing_raw, key):
        value = float(timing_raw.get(key, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} timing must be finite and nonnegative")
        return value
