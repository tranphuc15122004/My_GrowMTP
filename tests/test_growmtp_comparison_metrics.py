import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "verl" / "verl" / "trainer" / "mtp" / "metrics.py"
COMPARE_PATH = REPO_ROOT / "scripts" / "compare_growmtp_metrics.py"


def _load_module(path, name):
    assert path.exists(), f"Missing comparison utility: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _metrics():
    return _load_module(METRICS_PATH, "growmtp_comparison_metrics")


def _comparison():
    return _load_module(COMPARE_PATH, "growmtp_compare_script")


def _write_log(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_trajectory_means_ignore_padding_and_preserve_empty_trajectories():
    metrics = _metrics()
    advantages = torch.tensor([[2.0, 4.0, float("nan")], [-3.0, 999.0, 999.0], [5.0, 6.0, 7.0]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0]])

    result = metrics.trajectory_advantages(advantages, mask)

    torch.testing.assert_close(result[:2], torch.tensor([3.0, -3.0]))
    assert result.shape == (3,)
    assert torch.isnan(result[2])


def test_trajectory_statistics_weight_trajectories_and_exclude_empty_masks():
    metrics = _metrics()
    advantages = torch.tensor([[3.0, 3.0], [-3.0, 999.0], [0.0, 999.0], [99.0, 99.0]])
    mask = torch.tensor([[1, 1], [1, 0], [1, 0], [0, 0]])

    result = metrics.trajectory_statistics(advantages, mask)

    assert result["comparison/advantage/mean"] == 0.0
    assert result["comparison/advantage/std"] == pytest.approx(math.sqrt(6.0))
    for sign in ("positive", "negative", "zero"):
        assert result[f"comparison/advantage/{sign}_fraction"] == pytest.approx(1 / 3)
    assert result["comparison/advantage/valid_trajectories"] == 3
    assert result["comparison/advantage/empty_trajectories"] == 1
    assert result["comparison/advantage/total_trajectories"] == 4


def test_all_empty_masks_do_not_claim_zero_advantages():
    metrics = _metrics()
    result = metrics.trajectory_statistics(torch.ones(2, 3), torch.zeros(2, 3))

    assert result["comparison/advantage/valid_trajectories"] == 0
    assert result["comparison/advantage/empty_trajectories"] == 2
    assert math.isnan(result["comparison/advantage/mean"])
    assert math.isnan(result["comparison/advantage/zero_fraction"])


def test_nested_trajectory_means_match_padded_data():
    metrics = _metrics()
    advantages = torch.nested.nested_tensor([torch.tensor([2.0, 4.0]), torch.tensor([-3.0])], layout=torch.jagged)
    mask = torch.nested.nested_tensor([torch.tensor([1, 1]), torch.tensor([1])], layout=torch.jagged)

    torch.testing.assert_close(metrics.trajectory_advantages(advantages, mask), torch.tensor([3.0, -3.0]))


def test_trajectory_shape_mismatch_is_rejected():
    metrics = _metrics()
    with pytest.raises(ValueError, match="shape|length|trajector"):
        metrics.trajectory_advantages(torch.ones(2, 3), torch.ones(2, 2))


def test_nonfinite_token_advantages_are_counted_separately_from_empty_masks():
    metrics = _metrics()
    result = metrics.trajectory_statistics(torch.tensor([[float("inf")], [0.0], [1.0]]),
                                          torch.tensor([[1], [0], [1]]))

    assert result["comparison/advantage/nonfinite_trajectories"] == 1
    assert result["comparison/advantage/empty_trajectories"] == 1
    assert result["comparison/advantage/valid_trajectories"] == 1
    assert result["comparison/advantage/mean"] == 1.0
    assert result["comparison/advantage/std"] == 0.0


def test_tracker_counts_probe_once_and_includes_testing_in_observed_e2e():
    metrics = _metrics()
    tracker = metrics.ComparisonTracker()
    first = tracker.update({"step": 10.0, "mtp_probe": 2.0, "testing": 3.0}, n_gpus=2, response_tokens=100)
    second = tracker.update({"step": 5.0, "probe": 1.0}, n_gpus=2, response_tokens=50)

    assert first["comparison/time/step_training_s"] == 8.0
    assert first["comparison/time/step_e2e_s"] == 13.0
    assert second["comparison/time/session_training_s"] == 12.0
    assert second["comparison/time/session_evaluation_s"] == 3.0
    assert second["comparison/time/session_probe_s"] == 3.0
    assert second["comparison/time/session_e2e_s"] == 18.0
    assert second["comparison/time/session_gpu_hours"] == pytest.approx(0.01)
    assert second["comparison/tokens/session_generated"] == 150
    assert metrics.ComparisonTracker().update({"step": 1.0}, 1, 10)["comparison/time/session_training_s"] == 1.0


def test_tracker_accepts_initial_evaluation_without_a_training_step():
    metrics = _metrics()
    result = metrics.ComparisonTracker().update({"testing": 4.0}, n_gpus=2, response_tokens=0)

    assert result["comparison/time/session_training_s"] == 0.0
    assert result["comparison/time/session_evaluation_s"] == 4.0
    assert result["comparison/time/session_e2e_s"] == 4.0
    assert "comparison/time/step_training_s" not in result


@pytest.mark.parametrize("timing", [{"step": 1, "probe": 2}, {"step": -1}, {"step": 1, "probe": 0.1, "mtp_probe": 0.1}])
def test_tracker_rejects_ambiguous_or_impossible_timings(timing):
    metrics = _metrics()
    with pytest.raises(ValueError):
        metrics.ComparisonTracker().update(timing, n_gpus=1, response_tokens=1)


def test_summary_deduplicates_resumed_steps_and_preserves_validation_only_records(tmp_path):
    comparison = _comparison()
    path = _write_log(tmp_path / "metrics.jsonl", [
        {"step": 0, "val-core/math/reward/mean@1": 0.1, "timing_s/testing": 4},
        {"step": 1, "timing_s/step": 10, "critic/score/mean": 0.2, "perf/num_response_tokens": 100,
         "rollout/mtp/accepted_draft_tokens": 6, "rollout/mtp/proposed_draft_tokens": 10,
         "rollout/mtp/verification_steps": 2, "perf/throughput": 900,
         "perf/rollout_tokens_per_second_per_gpu": 50},
        {"step": 2, "timing_s/step": 999, "critic/score/mean": -5},
        {"step": 2, "timing_s/step": 20, "timing_s/testing": 5, "critic/score/mean": 0.8,
         "perf/num_response_tokens": 200, "rollout/mtp/accepted_draft_tokens": 2,
         "rollout/mtp/proposed_draft_tokens": 10, "rollout/mtp/verification_steps": 2},
        {"step": 2, "val-core/math/reward/mean@1": 0.7},
    ])

    summary = comparison.summarize_logs([path])

    assert summary["training"]["steps"] == 2
    assert summary["time"]["training_s"] == 30
    assert summary["time"]["evaluation_s"] == 9
    assert summary["time"]["e2e_s"] == 39
    assert summary["time"]["probe_s"] is None
    assert summary["acceptance"]["rate"] == pytest.approx(0.4)
    assert summary["acceptance"]["length"] == pytest.approx(3.0)
    assert summary["acceptance"]["aggregation"] == "pooled_counts"
    assert summary["reward"]["mean"] == pytest.approx(0.5)
    assert summary["reward"]["last"] == 0.8
    assert summary["validation"]["val-core/math/reward/mean@1"]["last"] == 0.7
    assert summary["throughput"]["training_tokens_per_second_per_gpu"]["mean"] == 900
    assert summary["throughput"]["rollout_tokens_per_second_per_gpu"]["mean"] == 50
    assert summary["coverage"]["duplicate_training_records"] == 1
    assert summary["coverage"]["validation_only_records"] == 2


def test_summary_reports_missing_metrics_and_unweighted_acceptance_explicitly(tmp_path):
    comparison = _comparison()
    path = _write_log(tmp_path / "old.jsonl", [
        {"step": 1, "timing_s/step": 10, "rollout/mtp/acceptance_length": 2},
        {"step": 2, "timing_s/step": 20, "rollout/mtp/acceptance_length": 4},
    ])

    summary = comparison.summarize_logs([path])

    assert summary["acceptance"]["length"] == 3
    assert summary["acceptance"]["aggregation"] == "step_mean_without_counts"
    assert summary["acceptance"]["rate"] is None
    assert summary["reward"]["mean"] is None
    assert summary["time"]["evaluation_s"] is None
    assert summary["time"]["gpu_hours"] is None
    assert summary["tokens"]["generated"] is None
    assert summary["coverage"]["reward_steps"] == 0


def test_summary_merges_sessions_from_per_step_metrics_and_never_sums_session_totals(tmp_path):
    comparison = _comparison()
    first = _write_log(tmp_path / "first.jsonl", [
        {"step": 1, "comparison/time/step_training_s": 8, "comparison/time/step_probe_s": 2,
         "comparison/time/step_evaluation_s": 3, "comparison/time/step_e2e_s": 13,
         "comparison/time/session_e2e_s": 13, "comparison/time/step_gpu_hours": 0.01,
         "comparison/tokens/step_generated": 100, "comparison/schema_version": 1},
    ])
    resumed = _write_log(tmp_path / "resumed.jsonl", [
        {"step": 2, "comparison/time/step_training_s": 4, "comparison/time/step_probe_s": 1,
         "comparison/time/step_evaluation_s": 0, "comparison/time/step_e2e_s": 5,
         "comparison/time/session_e2e_s": 5, "comparison/time/step_gpu_hours": 0.005,
         "comparison/tokens/step_generated": 50, "comparison/schema_version": 1},
    ])

    summary = comparison.summarize_logs([first, resumed])

    assert summary["time"]["training_s"] == 12
    assert summary["time"]["probe_s"] == 3
    assert summary["time"]["evaluation_s"] == 3
    assert summary["time"]["e2e_s"] == 18
    assert summary["time"]["gpu_hours"] == 0.015
    assert summary["tokens"]["generated"] == 150
    assert summary["schema_versions"] == [1]


def test_summary_includes_gpu_hours_from_initial_validation_and_nested_file_logger_rows(tmp_path):
    comparison = _comparison()
    path = _write_log(tmp_path / "metrics.jsonl", [
        {"step": 0, "data": {"val-core/math/reward/mean@1": 0.1,
                              "comparison/time/step_evaluation_s": 4,
                              "comparison/time/step_e2e_s": 4,
                              "comparison/time/step_gpu_hours": 0.002}},
        {"step": 1, "data": {"comparison/time/step_training_s": 10,
                              "comparison/time/step_evaluation_s": 0,
                              "comparison/time/step_probe_s": 0,
                              "comparison/time/step_e2e_s": 10,
                              "comparison/time/step_gpu_hours": 0.003}},
    ])

    summary = comparison.summarize_logs([path])

    assert summary["training"]["steps"] == 1
    assert summary["time"]["e2e_s"] == 14
    assert summary["time"]["gpu_hours"] == 0.005


@pytest.mark.parametrize("attached_evaluation", [0, 3])
def test_validation_only_event_at_training_step_adds_time_without_replacing_attached_evaluation(
    tmp_path, attached_evaluation
):
    comparison = _comparison()
    path = _write_log(tmp_path / "metrics.jsonl", [
        {"step": 7, "comparison/time/step_training_s": 10,
         "comparison/time/step_evaluation_s": attached_evaluation,
         "comparison/time/step_e2e_s": 10 + attached_evaluation,
         "comparison/time/step_gpu_hours": (10 + attached_evaluation) / 3600},
        {"step": 7, "comparison/time/step_evaluation_s": 4,
         "comparison/time/step_e2e_s": 4, "comparison/time/step_gpu_hours": 4 / 3600,
         "comparison/evaluation/final": 1},
    ])

    summary = comparison.summarize_logs([path])

    assert summary["training"]["steps"] == 1
    assert summary["time"]["evaluation_s"] == 4 + attached_evaluation
    assert summary["time"]["e2e_s"] == 14 + attached_evaluation
    assert summary["time"]["gpu_hours"] == pytest.approx((14 + attached_evaluation) / 3600)
    assert summary["coverage"]["validation_only_records"] == 1
    assert summary["coverage"]["standalone_evaluation_events"] == 1


def test_repeated_resume_evaluations_at_same_step_are_all_counted_and_training_is_deduplicated(tmp_path):
    comparison = _comparison()
    first = _write_log(tmp_path / "first.jsonl", [
        {"step": 7, "timing_s/step": 20, "timing_s/testing": 1, "comparison/config/n_gpus": 2},
        {"step": 7, "timing_s/testing": 4, "comparison/config/n_gpus": 2},
    ])
    resumed = _write_log(tmp_path / "resumed.jsonl", [
        {"step": 7, "comparison/time/step_e2e_s": 6, "comparison/config/n_gpus": 2},
        {"step": 7, "timing_s/step": 10, "timing_s/testing": 2, "comparison/config/n_gpus": 2},
    ])

    summary = comparison.summarize_logs([first, resumed])

    assert summary["training"]["steps"] == 1
    assert summary["time"]["training_s"] == 10
    assert summary["time"]["evaluation_s"] == 12
    assert summary["time"]["e2e_s"] == 22
    assert summary["time"]["gpu_hours"] == pytest.approx(44 / 3600)
    assert summary["coverage"]["duplicate_training_records"] == 1
    assert summary["coverage"]["standalone_evaluation_events"] == 2


@pytest.mark.parametrize("extra_candidate_evaluation", [False, True])
def test_common_step_speedup_excludes_separate_evaluations_and_run_total_requires_matching_schedule(
    tmp_path, extra_candidate_evaluation
):
    comparison = _comparison()
    baseline = _write_log(tmp_path / "baseline.jsonl", [
        {"step": 1, "comparison/time/step_training_s": 10,
         "comparison/time/step_evaluation_s": 3, "comparison/time/step_e2e_s": 13},
        {"step": 1, "comparison/time/step_evaluation_s": 4, "comparison/time/step_e2e_s": 4,
         "comparison/evaluation/final": 1},
    ])
    candidate_rows = [
        {"step": 1, "comparison/time/step_training_s": 5,
         "comparison/time/step_evaluation_s": 1.5, "comparison/time/step_e2e_s": 6.5},
        {"step": 1, "comparison/time/step_evaluation_s": 2, "comparison/time/step_e2e_s": 2,
         "comparison/evaluation/final": 1},
    ]
    if extra_candidate_evaluation:
        candidate_rows.append({"step": 1, "timing_s/testing": 1})
    candidate = _write_log(tmp_path / "candidate.jsonl", candidate_rows)

    run = comparison.compare_logs([baseline, candidate])["runs"][1]

    assert run["common_training_step_e2e_speedup_vs_baseline"] == 2.0
    assert run["run_total_e2e_speedup_vs_baseline"] == (None if extra_candidate_evaluation else 2.0)
    if extra_candidate_evaluation:
        assert any("evaluation schedules" in note for note in run["notes"])


def test_acceptance_scalars_are_weighted_by_verification_or_proposal_counts(tmp_path):
    comparison = _comparison()
    path = _write_log(tmp_path / "metrics.jsonl", [
        {"step": 1, "timing_s/step": 1, "draft/acceptance_length": 2,
         "draft/verification_steps": 10, "draft/acceptance_rate": 0.2, "draft/proposed_draft_tokens": 10},
        {"step": 2, "timing_s/step": 1, "draft/acceptance_length": 4,
         "draft/verification_steps": 30, "draft/acceptance_rate": 0.8, "draft/proposed_draft_tokens": 30},
    ])

    summary = comparison.summarize_logs([path])

    assert summary["acceptance"]["length"] == 3.5
    assert summary["acceptance"]["rate"] == pytest.approx(0.65)
    assert summary["acceptance"]["aggregation"] == "pooled_counts"


def test_summary_surfaces_probe_mechanism_and_head_update_time_without_refresh_zeros(tmp_path):
    comparison = _comparison()
    path = _write_log(tmp_path / "metrics.jsonl", [
        {"step": 1, "timing_s/step": 10, "draft/update_seconds": 0.1,
         "draft/probe/kl_new_old_mean": 0.2, "draft/probe/delta_lag_surrogate": -0.3},
        {"step": 2, "timing_s/step": 10, "draft/update_seconds": 0.3},
        {"step": 3, "timing_s/step": 10, "draft/probe/kl_new_old_mean": 0.4,
         "draft/probe/delta_lag_surrogate": -0.1},
    ])

    summary = comparison.summarize_logs([path])

    assert summary["probe"]["draft/probe/kl_new_old_mean"]["mean"] == pytest.approx(0.3)
    assert summary["probe"]["draft/probe/delta_lag_surrogate"]["mean"] == pytest.approx(-0.2)
    assert summary["draft_update_seconds"]["mean"] == pytest.approx(0.2)
    assert summary["coverage"]["draft_update_steps"] == 2
    assert summary["coverage"]["probe_metric_steps"]["draft/probe/kl_new_old_mean"] == 2
    assert not any("refresh" in key for key in summary["probe"])


def test_summary_reports_refresh_and_exact_shift_without_zero_filling_other_steps(tmp_path):
    comparison = _comparison()
    path = _write_log(tmp_path / "metrics.jsonl", [
        {"step": 3, "timing_s/step": 10},
        {"step": 4, "timing_s/step": 12, "draft/refresh/applied": 1,
         "draft/refresh/updated_cycles": 4, "draft/refresh/time_s": 2,
         "draft/shift/measured_trajectories": 5, "draft/shift/kl_new_old_mean": .2},
        {"step": 8, "timing_s/step": 11, "draft/refresh/applied": 0,
         "draft/refresh/updated_cycles": 0, "draft/refresh/time_s": 1,
         "draft/shift/measured_trajectories": 3, "draft/shift/kl_new_old_mean": .1},
    ])
    summary = comparison.summarize_logs([path])
    assert summary["refresh"]["draft/refresh/applied"]["mean"] == .5
    assert summary["refresh"]["draft/refresh/time_s"]["mean"] == 1.5
    assert summary["shift"]["draft/shift/kl_new_old_mean"]["records"] == 2
    assert summary["coverage"]["refresh_metric_steps"]["draft/refresh/updated_cycles"] == 2


def test_comparison_distinguishes_wall_time_speedup_from_gpu_cost(tmp_path):
    comparison = _comparison()
    baseline = _write_log(tmp_path / "baseline.jsonl", [
        {"step": 1, "timing_s/step": 10, "comparison/config/n_gpus": 2},
    ])
    candidate = _write_log(tmp_path / "candidate.jsonl", [
        {"step": 1, "timing_s/step": 5, "comparison/config/n_gpus": 4},
    ])

    report = comparison.compare_logs([baseline, candidate])

    run = report["runs"][1]
    assert run["e2e_speedup_vs_baseline"] == 2.0
    assert run["gpu_hours_speedup_vs_baseline"] == 1.0
    assert run["gpu_count_matches_baseline"] is False
    assert any("GPU count" in note for note in run["notes"])


def test_comparison_serializes_missing_or_nonfinite_values_as_null(tmp_path):
    comparison = _comparison()
    baseline = _write_log(tmp_path / "baseline.jsonl", [
        {"step": 1, "timing_s/step": 10, "critic/score/mean": float("nan")},
    ])
    candidate = _write_log(tmp_path / "candidate.jsonl", [
        {"step": 1, "timing_s/step": 1e-308, "draft/update_seconds": float("inf")},
    ])

    report = comparison.compare_logs([baseline, candidate])

    assert report["runs"][0]["reward"]["mean"] is None
    assert report["runs"][1]["draft_update_seconds"]["mean"] is None
    assert report["runs"][1]["e2e_speedup_vs_baseline"] is None
    assert "Infinity" not in json.dumps(report, allow_nan=False)


def test_cli_compares_e2e_on_common_training_steps_and_writes_strict_json(tmp_path):
    baseline = _write_log(tmp_path / "baseline.jsonl", [
        {"step": 1, "timing_s/step": 10, "timing_s/testing": 2},
        {"step": 2, "timing_s/step": 10, "timing_s/testing": 3},
        {"step": 3, "timing_s/step": 100},
    ])
    candidate = _write_log(tmp_path / "candidate.jsonl", [
        {"step": 1, "timing_s/step": 5, "timing_s/testing": 1},
        {"step": 2, "timing_s/step": 5, "timing_s/testing": 1.5},
    ])
    output = tmp_path / "comparison.json"

    result = subprocess.run([sys.executable, str(COMPARE_PATH), str(baseline), str(candidate),
                             "--output", str(output)], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["runs"][1]["e2e_speedup_vs_baseline"] == pytest.approx(2.0)
    assert report["runs"][1]["comparison_training_steps"] == 2
    assert report["runs"][1]["reward"]["mean"] is None
    assert "NaN" not in output.read_text()


def test_malformed_log_reports_file_and_line(tmp_path):
    comparison = _comparison()
    path = tmp_path / "broken.jsonl"
    path.write_text('{"step": 1}\nnot-json\n')

    with pytest.raises(ValueError, match="broken.jsonl:2"):
        comparison.summarize_logs([path])
