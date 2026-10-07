"""Small-data checks for the Vietnamese summarization GrowMTP workflow."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "scripts" / "prepare_vn_summarization.py"


def prepare(source: Path, destination: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PREPARE), "--input", str(source), "--output-dir", str(destination), *extra],
        text=True,
        capture_output=True,
        check=False,
    )


def test_sample_converts_to_verl_schema_without_split_leakage(tmp_path):
    destination = tmp_path / "prepared"
    result = prepare(ROOT / "data" / "sample.txt", destination, "--validation-fraction", "0.2")
    assert result.returncode == 0, result.stderr

    train = pq.read_table(destination / "train.parquet").to_pylist()
    validation = pq.read_table(destination / "validation.parquet").to_pylist()
    assert len(train) + len(validation) == 5
    assert train and validation
    assert {row["extra_info"]["id"] for row in train}.isdisjoint(
        {row["extra_info"]["id"] for row in validation}
    )
    assert {row["prompt"][0]["content"] for row in train}.isdisjoint(
        {row["prompt"][0]["content"] for row in validation}
    )
    for row in train + validation:
        assert set(row) == {"prompt", "data_source", "ability", "reward_model", "extra_info"}
        assert row["prompt"][0]["role"] == "user"
        assert "Tóm tắt" in row["prompt"][0]["content"]
        assert row["data_source"] == "vn_summarization"
        assert row["reward_model"]["style"] == "rule"
        assert row["reward_model"]["ground_truth"]
    assert pq.ParquetFile(destination / "train.parquet").metadata.num_row_groups > 0
    loaded = load_dataset("parquet", data_files=str(destination / "train.parquet"), split="train")
    assert len(loaded) == len(train)
    assert loaded[0]["reward_model"]["ground_truth"] == train[0]["reward_model"]["ground_truth"]
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["seed"] == 1


def test_pure_jsonl_parses_first_object_and_records_requested_seed(tmp_path):
    source = tmp_path / "train_clean.jsonl"
    records = [
        {"id": "first", "text": "Bài báo đầu tiên.", "summary": "Tóm tắt thứ nhất."},
        {"id": "second", "text": "Bài báo thứ hai.", "summary": "Tóm tắt thứ hai."},
    ]
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    destination = tmp_path / "prepared"
    result = prepare(source, destination, "--seed", "23", "--validation-fraction", "0.5")
    assert result.returncode == 0, result.stderr

    train = pq.read_table(destination / "train.parquet").to_pylist()
    validation = pq.read_table(destination / "validation.parquet").to_pylist()
    assert {row["extra_info"]["id"] for row in train + validation} == {"first", "second"}
    assert json.loads((destination / "manifest.json").read_text())["seed"] == 23


def test_prompt_audit_uses_chat_template_and_rejects_over_limit():
    module = runpy.run_path(str(PREPARE))
    audit_prompt_lengths = module.get("audit_prompt_lengths")
    convert_record = module["convert_record"]
    assert callable(audit_prompt_lengths), "prompt tokenizer audit is required before B200 training"

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking, return_dict):
            assert tokenize is True
            assert add_generation_prompt is True
            assert enable_thinking is False
            assert return_dict is False
            return list(range(len(messages[0]["content"].split()) + 3))

    rows = [convert_record({"id": "1", "text": "một hai ba", "summary": "tóm tắt"}, 1)]
    audit = audit_prompt_lengths(rows, FakeTokenizer(), max_prompt_length=1000)
    assert audit["prompt_count"] == 1
    assert audit["max_prompt_tokens"] == len(rows[0]["prompt"][0]["content"].split()) + 3
    assert audit["max_prompt_tokens"] <= audit["max_prompt_length"]

    with pytest.raises(ValueError, match="exceeds max_prompt_length"):
        audit_prompt_lengths(rows, FakeTokenizer(), max_prompt_length=4)


@pytest.fixture
def local_chat_tokenizer(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    return tokenizer, tmp_path / "tokenizer"


def test_prompt_audit_counts_token_ids_with_transformers_dictionary_default(local_chat_tokenizer):
    tokenizer, _ = local_chat_tokenizer
    module = runpy.run_path(str(PREPARE))
    rows = [module["convert_record"]({"id": "1", "text": "Bài báo tiếng Việt.", "summary": "Tóm tắt."}, 1)]
    expected_ids = tokenizer.apply_chat_template(
        rows[0]["prompt"], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    )
    audit = module["audit_prompt_lengths"](rows, tokenizer, max_prompt_length=1000)
    assert audit["max_prompt_tokens"] == len(expected_ids)
    assert audit["max_prompt_tokens"] > 2
    with pytest.raises(ValueError, match="exceeds max_prompt_length"):
        module["audit_prompt_lengths"](rows, tokenizer, max_prompt_length=3)


def _existing_audit_dataset(tmp_path, tokenizer_path):
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    module = runpy.run_path(str(PREPARE))
    for index, filename in enumerate(("train.parquet", "validation.parquet"), 1):
        row = module["convert_record"]({
            "id": str(index), "text": f"Bài báo tiếng Việt số {index}.", "summary": "Tóm tắt."
        }, index)
        pq.write_table(pa.Table.from_pylist([row], schema=module["SCHEMA"]), data_dir / filename)
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "seed": 23, "train_rows": 1, "validation_rows": 1,
        "prompt_audit": {"max_prompt_tokens": 2, "tokenizer": str(tokenizer_path)},
    }))
    return data_dir, manifest_path


@pytest.mark.parametrize("limit", [3, 1000])
def test_manifest_prompt_reaudit_preserves_parquet_and_rejects_overlong(
    tmp_path, local_chat_tokenizer, limit
):
    _, tokenizer_path = local_chat_tokenizer
    data_dir, manifest_path = _existing_audit_dataset(tmp_path, tokenizer_path)
    original_manifest = manifest_path.read_bytes()
    original_parquet = {name: (data_dir / name).read_bytes() for name in ("train.parquet", "validation.parquet")}
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts" / "validate_vn_summarization_data.py"),
        "--manifest", str(manifest_path), "--seed", "23", "--max-prompt-length", str(limit),
        "--refresh-prompt-audit", "--tokenizer", str(tokenizer_path),
    ], text=True, capture_output=True, check=False)
    assert all((data_dir / name).read_bytes() == content for name, content in original_parquet.items())
    if limit == 3:
        assert result.returncode != 0
        assert "exceeds max_prompt_length" in result.stderr
        assert manifest_path.read_bytes() == original_manifest
    else:
        assert result.returncode == 0, result.stderr
        manifest = json.loads(manifest_path.read_text())
        assert manifest["seed"] == 23
        assert manifest["train_rows"] == manifest["validation_rows"] == 1
        assert manifest["prompt_audit"]["max_prompt_tokens"] > 2
        assert manifest["prompt_audit"]["prompt_count"] == 2


def test_manifest_validator_rejects_seed_mismatch_and_missing_prompt_audit(tmp_path):
    validator = ROOT / "scripts" / "validate_vn_summarization_data.py"
    assert validator.is_file(), "server preflight must validate the data manifest before training"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "seed": 23,
        "train_rows": 9,
        "validation_rows": 1,
        "prompt_audit": {
            "max_prompt_tokens": 100,
            "max_prompt_length": 4096,
            "tokenizer": "qwen3-4b",
        },
    }))
    accepted = subprocess.run(
        [sys.executable, str(validator), "--manifest", str(manifest), "--seed", "23",
         "--max-prompt-length", "4096"], text=True, capture_output=True, check=False,
    )
    assert accepted.returncode == 0, accepted.stderr

    mismatch = subprocess.run(
        [sys.executable, str(validator), "--manifest", str(manifest), "--seed", "1",
         "--max-prompt-length", "4096"], text=True, capture_output=True, check=False,
    )
    assert mismatch.returncode != 0
    assert "seed" in mismatch.stderr.lower()

    manifest.write_text(json.dumps({"seed": 23}))
    missing_audit = subprocess.run(
        [sys.executable, str(validator), "--manifest", str(manifest), "--seed", "23",
         "--max-prompt-length", "4096"], text=True, capture_output=True, check=False,
    )
    assert missing_audit.returncode != 0
    assert "prompt audit" in missing_audit.stderr.lower()


def test_invalid_row_does_not_create_output(tmp_path):
    source = tmp_path / "invalid.jsonl"
    source.write_text(json.dumps({"id": "1", "text": "Văn bản", "summary": ""}) + "\n")
    destination = tmp_path / "prepared"
    result = prepare(source, destination)
    assert result.returncode != 0
    assert not destination.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    destination = tmp_path / "prepared"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("untouched")
    result = prepare(ROOT / "data" / "sample.txt", destination)
    assert result.returncode != 0
    assert marker.read_text() == "untouched"


def test_reward_is_bounded_and_has_learning_signal():
    sys.path.insert(0, str(ROOT / "scripts"))
    from vn_summarization_reward import compute_score

    reference = "Cô gái lo lắng khi gia đình phản đối mối quan hệ."
    assert compute_score("vn_summarization", reference, reference) == 1.0
    assert compute_score("vn_summarization", "Một con mèo đang ngủ.", reference) == 0.0
    partial = compute_score("vn_summarization", "Gia đình phản đối mối quan hệ.", reference)
    assert 0.0 < partial < 1.0
