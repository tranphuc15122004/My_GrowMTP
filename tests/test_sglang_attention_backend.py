import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "verl/verl/workers/rollout/sglang_rollout/attention_backend.py"
)


def load_attention_backend_policy():
    if not MODULE_PATH.is_file():
        raise AssertionError(
            "SGLang rollout needs a hardware-aware attention backend policy"
        )
    spec = importlib.util.spec_from_file_location(
        "growmtp_sglang_attention_backend", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SGLangAttentionBackendTests(unittest.TestCase):
    def test_blackwell_defaults_to_flashinfer(self):
        policy = load_attention_backend_policy()

        self.assertEqual(
            policy.resolve_sglang_attention_backend(None, (10, 0)), "flashinfer"
        )

    def test_older_gpus_keep_fa3_default(self):
        policy = load_attention_backend_policy()

        self.assertEqual(policy.resolve_sglang_attention_backend(None, (9, 0)), "fa3")

    def test_explicit_backend_is_preserved_on_blackwell(self):
        policy = load_attention_backend_policy()

        self.assertEqual(
            policy.resolve_sglang_attention_backend("triton", (10, 0)), "triton"
        )


if __name__ == "__main__":
    unittest.main()
