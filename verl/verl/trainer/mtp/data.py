"""Convert local Math or Code parquet files into veRL's rule-reward schema."""

import argparse
import json
from pathlib import Path


def normalize_code_tests(tests):
    """Normalize TACO function arguments into the evaluator's JSON-lines convention."""
    tests = json.loads(tests) if isinstance(tests, str) else dict(tests)
    if tests.get("fn_name"):
        tests["inputs"] = [
            item if isinstance(item, str) else "\n".join(json.dumps(arg) for arg in item)
            for item in tests["inputs"]
        ]

        def encoded(value):
            if isinstance(value, str):
                try:
                    json.loads(value)
                    return value
                except json.JSONDecodeError:
                    pass
            return json.dumps(value)

        tests["outputs"] = [encoded(item) for item in tests["outputs"]]
    return tests


def convert_row(row, task, index):
    prompt = row.get("prompt", row.get("source_prompt"))
    if prompt is None:
        prompt = row.get("question", row.get("problem"))
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    elif hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"Row {index}: missing prompt/question/problem")
    reward = row.get("reward_model")
    if not isinstance(reward, dict) or "ground_truth" not in reward:
        if task == "math":
            answer = row.get("answer", row.get("solution"))
            if answer is None:
                raise ValueError(f"Row {index}: missing reference answer")
            reward = {"style": "rule", "ground_truth": str(answer)}
        else:
            tests = row.get("input_output")
            if tests is None:
                raise ValueError(f"Row {index}: missing code input_output tests")
            if isinstance(tests, str):
                tests = json.loads(tests)
            if "inputs" not in tests or "outputs" not in tests:
                raise ValueError(f"Row {index}: code tests need inputs and outputs")
            reward = {"style": "rule", "ground_truth": json.dumps(normalize_code_tests(tests))}
    return {
        "prompt": prompt,
        "data_source": "math_dapo" if task == "math" else "taco",
        "ability": task,
        "reward_model": reward,
        "extra_info": {"index": index},
    }


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["math", "code"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError("Data output already exists")
    data = pd.read_parquet(args.input)
    result = [convert_row(row, args.task, i) for i, row in enumerate(data.to_dict("records"))]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result).to_parquet(args.output, index=False)
    print(f"Converted {len(result)} rows; no filtering or subsampling.")


if __name__ == "__main__":
    main()
