"""Trajectory selection and teacher replacement; DCA/VGM stays in engine/replay.

Full-vocabulary shift detection streams old FP32 log probabilities to a temporary
file on DP rank zero. Every FSDP rank executes the same ordered verifier forwards.
Only selected trajectories receive new hidden states and top-k supervision.
"""

from contextlib import closing
import math
import os
import tempfile

import torch
import torch.distributed as dist

from .probe import cycle_case, iter_evaluations
from .signals import FIELDS


def trajectory_key(record):
    return record["source_rank"], record["trajectory_index"]


def gather_trajectories(records, data, group=None):
    # Do not all-gather the large rollout hidden states or teacher tensors.
    compact = [{key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for key, value in record.items() if key != "prefix_hidden"} for record in records]
    part = (compact, len(data["mtp_num_cycles"]), int(data["mtp_num_cycles"].sum()))
    parts = [part]
    if dist.is_initialized():
        parts = [None] * dist.get_world_size(group)
        dist.all_gather_object(parts, part, group=group)
    records = sorted([row for rows, _, _ in parts for row in rows], key=trajectory_key)
    return records, sum(n for _, n, _ in parts), sum(n for _, _, n in parts)


def record_cases(records):
    for record in records:
        for cycle in range(record["position"].numel()):
            yield cycle_case(record, cycle)


def capture_shift(model, records, *, source):
    """Raw teacher convention matches SGLang, independently of rollout temperature."""
    cache_dir = os.environ.get("GROWMTP_SHIFT_CACHE_DIR")
    cache = tempfile.TemporaryFile(dir=cache_dir) if source else None
    positive = [record for record in records if record["advantage"] > 0]
    positions = 0
    try:
        if positive:
            with closing(iter_evaluations(model, record_cases(positive))) as evaluations:
                for target, _, _ in evaluations:
                    if not torch.isfinite(target).all():
                        raise ValueError("Non-finite old target distribution in policy-shift detector")
                    positions += target.shape[0]
                    if cache is not None:
                        cache.write(target.contiguous().numpy().tobytes())
        size = cache.tell() if cache is not None else 0
        return {"cache": cache, "positive": positive, "num_positions": positions, "cache_bytes": size}
    except BaseException:
        if cache is not None:
            cache.close()
        raise


def score_shift(model, records, snapshot, group=None):
    """D_r averages all verifier positions in all recorded cycles of trajectory r."""
    cache = snapshot["cache"]
    scores = {trajectory_key(row): 0.0 for row in records}
    if cache is not None:
        cache.seek(0)
    try:
        positive = snapshot["positive"]
        if positive:
            with closing(iter_evaluations(model, record_cases(positive))) as evaluations:
                for record in positive:
                    total, count = 0.0, 0
                    for _ in range(record["position"].numel()):
                        new, _, _ = next(evaluations)
                        if not torch.isfinite(new).all():
                            raise ValueError("Non-finite updated target distribution in policy-shift detector")
                        if cache is not None:
                            raw = bytearray(cache.read(new.numel() * new.element_size()))
                            if len(raw) != new.numel() * new.element_size():
                                raise RuntimeError("Incomplete policy-shift snapshot")
                            old = torch.frombuffer(raw, dtype=torch.float32).reshape(new.shape)
                            total += (new.exp() * (new - old)).sum(-1).clamp_min(0).sum().item()
                            count += new.shape[0]
                    if cache is not None:
                        scores[trajectory_key(record)] = total / max(1, count)
        if dist.is_initialized():
            payload = [scores, snapshot["cache_bytes"]]
            root = dist.get_global_rank(group, 0) if group is not None else 0
            dist.broadcast_object_list(payload, src=root, group=group)
            scores, snapshot["cache_bytes"] = payload
        return scores
    finally:
        if cache is not None:
            cache.close()


def select_trajectories(records, shift, fraction):
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("comparison_refresh_fraction must be in [0, 1]")
    scores = {trajectory_key(row): max(row["advantage"], 0.0) * shift[trajectory_key(row)]
              for row in records}
    positive = [row for row in records if scores[trajectory_key(row)] > 0]
    count = min(len(positive), math.ceil(fraction * len(records)))
    selected = sorted(positive, key=lambda row: (-scores[trajectory_key(row)], *trajectory_key(row)))[:count]
    return selected, scores


def mixed_teacher_batch(model, data, selected, source_rank, teacher_topk):
    """Preserve paths/VGM, replacing only teacher and target features on selected rows."""
    replacements = {}
    if selected:
        with closing(iter_evaluations(model, record_cases(selected), include_hidden=True)) as evaluations:
            for record in selected:
                values, indices = [], []
                for cycle in range(record["position"].numel()):
                    target, _, hidden = next(evaluations)
                    topk_val, topk_idx = target.topk(min(teacher_topk, target.shape[-1]), dim=-1)
                    if record["source_rank"] == source_rank:
                        values.append(topk_val)
                        indices.append(topk_idx)
                if record["source_rank"] == source_rank:
                    # The last verifier context contains the full committed prefix.
                    prefix = hidden[:int(record["position"][-1]) + 1]
                    replacements[record["trajectory_index"]] = {
                        "topk_val": torch.stack(values), "topk_idx": torch.stack(indices),
                        "prefix_hidden": prefix, "hidden": prefix[record["position"]],
                    }
    batch = {"mtp_num_cycles": data["mtp_num_cycles"]}
    for field in FIELDS:
        if "mtp_" + field not in data:
            continue
        original = data["mtp_" + field]
        if replacements and field in ("topk_val", "topk_idx", "hidden", "prefix_hidden"):
            rows = []
            for i, old in enumerate(original.unbind()):
                replacement = replacements.get(int(data["comparison_request_index"][i]))
                rows.append(replacement[field].to(old) if replacement else old)
            batch["mtp_" + field] = torch.nested.nested_tensor(rows, layout=torch.jagged)
        else:
            batch["mtp_" + field] = original
    return batch
