"""Reference-based ROUGE-L F1 reward for a Vietnamese summarization RL pilot."""

import re
import unicodedata


def _words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"\w+", normalized, flags=re.UNICODE)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0] * (len(right) + 1)
        for index, reference_token in enumerate(right, 1):
            current[index] = previous[index - 1] + 1 if token == reference_token else max(
                previous[index], current[index - 1]
            )
        previous = current
    return previous[-1]


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None, **kwargs) -> float:
    if data_source != "vn_summarization":
        raise ValueError(f"Unexpected data_source for summarization reward: {data_source}")
    generated = _words(solution_str)
    reference = _words(ground_truth)
    if not generated or not reference:
        return 0.0
    overlap = _lcs_length(generated, reference)
    return 2.0 * overlap / (len(generated) + len(reference))
