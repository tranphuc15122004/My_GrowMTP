#!/usr/bin/env python3
"""Summarize scalar JSONL logs and compare observed E2E on common steps.

Each positional file is a separate run. ``--merge`` instead combines resume
segments into one summary; later training records replace earlier records for
the same global step. Separate validation events retain each occurrence even
when several events share a training step. Session counters are never summed.
Legacy logs remain usable, with missing counters represented as null and
coverage reported. E2E covers measured training steps and evaluation, not
startup or other unmeasured process time. Equal step numbers alone do not prove
that datasets, batch sizes, hardware, or evaluation schedules match.
"""

import argparse
import json
import math
from pathlib import Path


VALIDATION_PREFIXES = ("val-core/", "val-aux/", "val/", "validation/")
TRAINING_KEYS = ("comparison/time/step_training_s", "timing_s/step", "perf/time_per_step",
                 "actor/pg_loss", "target/pg_loss", "critic/score/mean", "critic/rewards/mean")
ACCEPTED_KEYS = ("rollout/mtp/accepted_draft_tokens", "draft/accepted_draft_tokens")
PROPOSED_KEYS = ("rollout/mtp/proposed_draft_tokens", "draft/proposed_draft_tokens")
VERIFY_KEYS = ("rollout/mtp/verification_steps", "draft/verification_steps")


def _number(row, *keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    return None


def _json_safe(value):
    """Keep derived overflow/nonfinite results out of JSON reports too."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sum(values):
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _stats(values):
    values = [value for value in values if value is not None]
    return {"mean": sum(values) / len(values) if values else None,
            "first": values[0] if values else None, "last": values[-1] if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "records": len(values)}


def _read_logs(paths):
    training, validation, standalone_evaluations, configs = {}, {}, [], {}
    schemas = set()
    coverage = {"records": 0, "duplicate_training_records": 0, "validation_only_records": 0,
                "missing_step_records": 0, "standalone_evaluation_missing_timing_records": 0}
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected a JSON object")
                    if isinstance(row.get("data"), dict):
                        row = {**row["data"], "step": row.get("step")}
                    step = _number(row, "step")
                    if step is not None and (step < 0 or not step.is_integer()):
                        raise ValueError("step must be a nonnegative integer")
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                coverage["records"] += 1
                schema = _number(row, "comparison/schema_version")
                if schema is not None:
                    schemas.add(schema)
                configs.update({key: value for key, value in row.items()
                                if key.startswith("comparison/config/") and _number(row, key) is not None})
                if step is None:
                    coverage["missing_step_records"] += 1
                    continue
                step = int(step)
                is_training = any(key in row for key in TRAINING_KEYS)
                if is_training:
                    coverage["duplicate_training_records"] += int(step in training)
                    training[step] = row
                scores = {key: value for key, value in row.items()
                          if key.startswith(VALIDATION_PREFIXES) and _number(row, key) is not None}
                if scores:
                    validation.setdefault(step, {}).update(scores)
                evaluation_time = _number(row, "comparison/time/step_evaluation_s", "timing_s/testing")
                standalone_e2e = _number(row, "comparison/time/step_e2e_s")
                phase = ("initial" if _number(row, "comparison/evaluation/initial") == 1 else
                         "final" if _number(row, "comparison/evaluation/final") == 1 else "validation")
                if not is_training and (scores or evaluation_time is not None or standalone_e2e is not None or
                                        phase != "validation"):
                    coverage["validation_only_records"] += 1
                    if evaluation_time is None:
                        evaluation_time = standalone_e2e
                    if standalone_e2e is None:
                        standalone_e2e = evaluation_time
                    gpu_hours = _number(row, "comparison/time/step_gpu_hours")
                    n_gpus = _number(row, "comparison/config/n_gpus")
                    if gpu_hours is None and n_gpus is not None and standalone_e2e is not None:
                        gpu_hours = standalone_e2e * n_gpus / 3600.0
                    standalone_evaluations.append(_json_safe({
                        "step": step, "phase": phase, "evaluation": evaluation_time,
                        "e2e": standalone_e2e, "gpu_hours": gpu_hours, "n_gpus": n_gpus,
                    }))
                    coverage["standalone_evaluation_missing_timing_records"] += int(standalone_e2e is None)
    return training, validation, standalone_evaluations, configs, sorted(schemas), coverage


def _step_times(row):
    evaluation = _number(row, "comparison/time/step_evaluation_s", "timing_s/testing")
    raw_step = _number(row, "timing_s/step", "perf/time_per_step")
    training = _number(row, "comparison/time/step_training_s")
    probe = _number(row, "comparison/time/step_probe_s", "timing_s/mtp_probe", "timing_s/probe")
    if training is None and raw_step is not None:
        training = raw_step - (probe if probe is not None else 0.0)
    e2e = _number(row, "comparison/time/step_e2e_s")
    if e2e is None:
        enclosed_step = raw_step
        if enclosed_step is None and training is not None:
            enclosed_step = training + (probe if probe is not None else 0.0)
        if enclosed_step is not None:
            e2e = enclosed_step + (evaluation if evaluation is not None else 0.0)
    gpu_hours = _number(row, "comparison/time/step_gpu_hours")
    n_gpus = _number(row, "comparison/config/n_gpus")
    if gpu_hours is None and e2e is not None and n_gpus is not None:
        gpu_hours = e2e * n_gpus / 3600.0
    return _json_safe({"training": training, "probe": probe, "e2e": e2e, "evaluation": evaluation,
                       "gpu_hours": gpu_hours, "n_gpus": n_gpus})


def _acceptance(rows, metric, denominator_keys):
    """Pool accepted/denominator counts, or weight scalar means by counts."""
    numerator, denominator, weighted_steps = 0.0, 0.0, 0
    scalars = []
    scalar_keys = (f"rollout/mtp/acceptance_{metric}", f"draft/acceptance_{metric}")
    for row in rows:
        accepted = _number(row, *ACCEPTED_KEYS)
        count = _number(row, *denominator_keys)
        scalar = _number(row, *scalar_keys)
        if scalar is not None:
            scalars.append(scalar)
        if count is not None and count > 0 and (accepted is not None or scalar is not None):
            numerator += accepted if accepted is not None else (scalar - (metric == "length")) * count
            denominator += count
            weighted_steps += 1
    if denominator > 0:
        return numerator / denominator + (metric == "length"), "pooled_counts", weighted_steps
    if scalars:
        return sum(scalars) / len(scalars), "step_mean_without_counts", len(scalars)
    return None, None, 0


def summarize_logs(paths):
    """Combine one run's log segments, deduplicate global steps, and summarize."""
    paths = [Path(path) for path in paths]
    training, validation, standalone_evaluations, configs, schemas, coverage = _read_logs(paths)
    ordered = sorted(training)
    rows = [training[step] for step in ordered]
    times = {step: _step_times(training[step]) for step in ordered}
    rate, rate_method, rate_steps = _acceptance(rows, "rate", PROPOSED_KEYS)
    length, length_method, length_steps = _acceptance(rows, "length", VERIFY_KEYS)
    rewards = [_number(row, "critic/rewards/mean", "critic/score/mean") for row in rows]
    tokens = [_number(row, "comparison/tokens/step_generated", "perf/num_response_tokens") for row in rows]
    probe_keys = sorted({key for row in rows for key in row if key.startswith("draft/probe/")})
    probe_stats = {key: _stats([_number(row, key) for row in rows]) for key in probe_keys}
    refresh_stats = {key: _stats([_number(row, key) for row in rows]) for key in sorted({
        key for row in rows for key in row if key.startswith("draft/refresh/")})}
    shift_stats = {key: _stats([_number(row, key) for row in rows]) for key in sorted({
        key for row in rows for key in row if key.startswith("draft/shift/")})}
    draft_update_stats = _stats([_number(row, "draft/update_seconds") for row in rows])
    validation_stats = {}
    for step in sorted(validation):
        for key, value in validation[step].items():
            validation_stats.setdefault(key, []).append((step, value))
    for key, values in validation_stats.items():
        validation_stats[key] = {**_stats([value for _, value in values]),
                                 "first_step": values[0][0], "last_step": values[-1][0]}
    evaluations = ([value["evaluation"] for value in times.values()] +
                   [event["evaluation"] for event in standalone_evaluations])
    e2e = _sum([value["e2e"] for value in times.values()] + [event["e2e"] for event in standalone_evaluations])
    coverage.update({
        "training_time_steps": sum(value["training"] is not None for value in times.values()),
        "e2e_time_steps": sum(value["e2e"] is not None for value in times.values()),
        "probe_time_steps": sum(value["probe"] is not None for value in times.values()),
        "evaluation_time_records": sum(value is not None for value in evaluations),
        "standalone_evaluation_events": len(standalone_evaluations),
        "standalone_gpu_hours_events": sum(event["gpu_hours"] is not None for event in standalone_evaluations),
        "gpu_hours_training_steps": sum(value["gpu_hours"] is not None for value in times.values()),
        "generated_token_steps": sum(value is not None for value in tokens),
        "reward_steps": sum(value is not None for value in rewards),
        "acceptance_rate_steps": rate_steps, "acceptance_length_steps": length_steps,
        "draft_update_steps": draft_update_stats["records"],
        "probe_metric_steps": {key: stats["records"] for key, stats in probe_stats.items()},
        "refresh_metric_steps": {key: stats["records"] for key, stats in refresh_stats.items()},
        "shift_metric_steps": {key: stats["records"] for key, stats in shift_stats.items()},
    })
    throughput = {
        "training_tokens_per_second_per_gpu": _stats([_number(row, "perf/throughput") for row in rows]),
        "rollout_tokens_per_second_per_gpu": _stats([
            _number(row, "perf/rollout_tokens_per_second_per_gpu", "rollout/mtp/generated_tokens_per_second_per_gpu",
                    "draft/generated_tokens_per_second_per_gpu") for row in rows]),
    }
    notes = []
    if ordered and coverage["probe_time_steps"] < len(ordered):
        notes.append("Legacy step time may include unreported diagnostic probing.")
    if not any(value is not None for value in evaluations):
        notes.append("No evaluation timing was logged; E2E includes only observed step timing.")
    if coverage["standalone_evaluation_missing_timing_records"]:
        notes.append("Some separate validation events have no timing; run E2E covers only timed events.")
    if coverage["e2e_time_steps"] < len(ordered):
        notes.append("Time totals cover only training steps with timing metrics.")
    if rate_method == "step_mean_without_counts" or length_method == "step_mean_without_counts":
        notes.append("Acceptance without counts is an unweighted mean across logged steps.")
    return _json_safe({
        "sources": [str(path) for path in paths], "schema_versions": schemas, "config_scalars": configs,
        "training": {"steps": len(ordered), "first_step": ordered[0] if ordered else None,
                     "last_step": ordered[-1] if ordered else None},
        "time": {"training_s": _sum(value["training"] for value in times.values()),
                 "evaluation_s": _sum(evaluations), "probe_s": _sum(value["probe"] for value in times.values()),
                 "e2e_s": e2e,
                 "gpu_hours": _sum([value["gpu_hours"] for value in times.values()] +
                                   [event["gpu_hours"] for event in standalone_evaluations]),
                 "mean_training_step_e2e_s": _stats([value["e2e"] for value in times.values()])["mean"]},
        "tokens": {"generated": _sum(tokens)}, "reward": _stats(rewards), "validation": validation_stats,
        "acceptance": {"rate": rate, "length": length,
                       "aggregation": rate_method or length_method,
                       "rate_aggregation": rate_method, "length_aggregation": length_method},
        "throughput": throughput, "coverage": coverage, "notes": notes,
        "probe": probe_stats, "refresh": refresh_stats, "shift": shift_stats,
        "draft_update_seconds": draft_update_stats,
    })


def compare_logs(paths):
    """Compare common training steps; compare run totals only for matching schedules."""
    runs = [summarize_logs([path]) for path in paths]
    measured = []
    evaluation_schedules = []
    complete_timing = []
    for path in paths:
        training, _, standalone_evaluations, _, _, _ = _read_logs([path])
        times = {step: _step_times(row) for step, row in training.items()}
        measured.append(times)
        schedule = [("separate", event["step"], event["phase"]) for event in standalone_evaluations]
        schedule.extend(("attached", step, "final" if
                         _number(training[step], "comparison/evaluation/final") == 1 else "validation")
                        for step, timing in times.items() if timing["evaluation"] is not None and timing["evaluation"] > 0)
        evaluation_schedules.append(sorted(schedule))
        complete_timing.append(bool(times) and all(timing["e2e"] is not None and timing["evaluation"] is not None
                                                   for timing in times.values()) and
                               all(event["e2e"] is not None for event in standalone_evaluations))
    for index, (run, values) in enumerate(zip(runs, measured, strict=True)):
        common = sorted(step for step in measured[0].keys() & values.keys()
                        if measured[0][step]["e2e"] is not None and values[step]["e2e"] is not None)
        baseline_time = sum(measured[0][step]["e2e"] for step in common)
        candidate_time = sum(values[step]["e2e"] for step in common)
        run["comparison_training_steps"] = len(common)
        common_step_speedup = baseline_time / candidate_time if candidate_time > 0 else None
        run["common_training_step_e2e_speedup_vs_baseline"] = common_step_speedup
        # Retain the initial schema's field as an explicitly scoped alias.
        run["e2e_speedup_vs_baseline"] = common_step_speedup
        run["e2e_speedup_scope"] = "common_training_steps_with_attached_evaluation"
        run["comparison_baseline_e2e_s"] = baseline_time if common else None
        run["comparison_run_e2e_s"] = candidate_time if common else None
        gpu_counts_known = bool(common) and all(measured[0][step]["n_gpus"] is not None and
                                                values[step]["n_gpus"] is not None for step in common)
        run["gpu_count_matches_baseline"] = (all(measured[0][step]["n_gpus"] == values[step]["n_gpus"]
                                                 for step in common) if gpu_counts_known else None)
        gpu_hours_known = bool(common) and all(measured[0][step]["gpu_hours"] is not None and
                                               values[step]["gpu_hours"] is not None for step in common)
        gpu_hours_speedup = None
        if gpu_hours_known:
            baseline_gpu_hours = sum(measured[0][step]["gpu_hours"] for step in common)
            candidate_gpu_hours = sum(values[step]["gpu_hours"] for step in common)
            if candidate_gpu_hours > 0:
                gpu_hours_speedup = baseline_gpu_hours / candidate_gpu_hours
        run["gpu_hours_speedup_vs_baseline"] = gpu_hours_speedup
        run["common_training_step_gpu_hours_speedup_vs_baseline"] = gpu_hours_speedup
        if run["gpu_count_matches_baseline"] is False:
            run["notes"].append("GPU count differs from baseline; wall-clock speedup also reflects hardware allocation.")
        matching_steps = measured[0].keys() == values.keys()
        matching_schedule = evaluation_schedules[0] == evaluation_schedules[index]
        run_total_comparable = (matching_steps and matching_schedule and complete_timing[0] and complete_timing[index]
                                and runs[0]["time"]["e2e_s"] is not None and run["time"]["e2e_s"] is not None)
        run["run_total_timing_comparable"] = run_total_comparable
        run["run_total_e2e_speedup_vs_baseline"] = None
        if run_total_comparable and run["time"]["e2e_s"] is not None and run["time"]["e2e_s"] > 0:
            run["run_total_e2e_speedup_vs_baseline"] = runs[0]["time"]["e2e_s"] / run["time"]["e2e_s"]
        elif not run_total_comparable:
            run["notes"].append("Run-total speedup is unavailable: full training-step coverage and timed evaluation "
                                "schedules must match; common-step speedup excludes separate validation events.")
    return _json_safe({"schema_version": 1, "baseline": str(paths[0]), "runs": runs,
            "comparison_basis": "Common-step speedup uses shared global training steps and their attached evaluation. "
                                "Run-total speedup includes every separately logged validation event and requires "
                                "matching step coverage and evaluation schedules with complete timing. "
                                "Check dataset, batch size, GPU count, and evaluation schedule before interpreting speedup."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="metrics.jsonl files")
    parser.add_argument("--output", type=Path, help="Also write the report as JSON")
    parser.add_argument("--merge", action="store_true", help="Treat all files as resumed segments of one run")
    args = parser.parse_args()
    try:
        report = ({"schema_version": 1, "runs": [summarize_logs(args.logs)]}
                  if args.merge else compare_logs(args.logs))
        rendered = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Comparison failed: {error}\n")


if __name__ == "__main__":
    main()
