"""Paper task rewards: reference-answer string matching and local code tests."""

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.math_dapo import compute_score as math_score
from verl.utils.reward_score.math_dapo import last_boxed_only_string


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source in ("math_dapo", "math", "math_dapo_reasoning") or data_source.startswith(
        "aime"
    ):
        # Released math prompts ask for a boxed answer. Also accept the upstream
        # "Answer:" convention when there is no box; both are string matching.
        return math_score(
            solution_str,
            ground_truth,
            strict_box_verify=last_boxed_only_string(solution_str) is not None,
        )
    if data_source == "taco":
        from verl.trainer.mtp.data import normalize_code_tests

        ground_truth = normalize_code_tests(ground_truth)
    return default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info)
