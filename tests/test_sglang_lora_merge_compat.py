import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "verl/verl/workers/rollout/sglang_rollout/lora_compat.py"
)


def load_lora_compat():
    if not MODULE_PATH.is_file():
        raise AssertionError(
            "SGLang rollout needs a dependency-light LoRA compatibility policy"
        )
    spec = importlib.util.spec_from_file_location("growmtp_lora_compat", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_model_config(*, rank, merge, target_modules=("q_proj", "v_proj")):
    return SimpleNamespace(
        lora_rank=rank,
        target_modules=list(target_modules),
        lora={"merge": merge},
    )


class SGLangLoRAMergeCompatibilityTests(unittest.TestCase):
    def test_merged_lora_leaves_eagle_server_as_a_regular_model(self):
        compat = load_lora_compat()
        args = {"speculative_algorithm": "EAGLE", "growmtp_collect_signals": True}

        result = compat.configure_lora_server_args(
            args, make_model_config(rank=16, merge=True)
        )

        self.assertIs(result, args)
        self.assertEqual(
            result,
            {"speculative_algorithm": "EAGLE", "growmtp_collect_signals": True},
        )

    def test_unmerged_lora_still_registers_sglang_adapter(self):
        compat = load_lora_compat()
        args = {"speculative_algorithm": None}

        result = compat.configure_lora_server_args(
            args, make_model_config(rank=16, merge=False)
        )

        self.assertEqual(
            result,
            {
                "speculative_algorithm": None,
                "enable_lora": True,
                "max_lora_rank": 16,
                "lora_target_modules": ["q_proj", "v_proj"],
            },
        )

    def test_zero_rank_does_not_register_lora_adapter(self):
        compat = load_lora_compat()
        args = {"speculative_algorithm": "EAGLE"}

        result = compat.configure_lora_server_args(
            args, make_model_config(rank=0, merge=False)
        )

        self.assertEqual(result, {"speculative_algorithm": "EAGLE"})

    def test_nested_lora_rank_is_used_when_top_level_rank_is_zero(self):
        compat = load_lora_compat()
        config = make_model_config(rank=0, merge=False)
        config.lora["rank"] = 8
        args = {}

        result = compat.configure_lora_server_args(args, config)

        self.assertEqual(
            result,
            {
                "enable_lora": True,
                "max_lora_rank": 8,
                "lora_target_modules": ["q_proj", "v_proj"],
            },
        )

    def test_merged_lora_request_does_not_name_an_sglang_adapter(self):
        compat = load_lora_compat()

        self.assertIsNone(compat.lora_adapter_name(make_model_config(rank=16, merge=True)))
        self.assertEqual(
            compat.lora_adapter_name(make_model_config(rank=16, merge=False)),
            "verl_actor_lora_name",
        )


if __name__ == "__main__":
    unittest.main()
