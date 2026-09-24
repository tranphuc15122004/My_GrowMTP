import unittest
from pathlib import Path

from scripts.check_training_imports import verify_training_imports
from scripts.install_modal_image_deps import modal_runtime_dependencies
from scripts.run_modal_growmtp_smoke import MODAL_GPU, build_smoke_env


class ModalSmokeEnvironmentTests(unittest.TestCase):
    def test_build_smoke_env_overrides_b200_defaults_and_paths(self):
        env = build_smoke_env(
            python_executable="/usr/bin/python3",
            train_file="/data/dapo-train.parquet",
            validation_file="/data/dapo-validation.parquet",
            prepared_model_dir="/tmp/growmtp-prepared",
            run_output_dir="/tmp/growmtp-output",
        )

        self.assertEqual(env["GROWMTP_PYTHON"], "/usr/bin/python3")
        self.assertEqual(env["TRAIN_FILE"], "/data/dapo-train.parquet")
        self.assertEqual(env["VAL_FILE"], "/data/dapo-validation.parquet")
        self.assertEqual(env["BASE_MODEL"], "Qwen/Qwen3-4B")
        self.assertEqual(env["PREPARED_MODEL_DIR"], "/tmp/growmtp-prepared")
        self.assertEqual(env["RUN_OUTPUT_DIR"], "/tmp/growmtp-output")
        self.assertEqual(env["RUN_MODE"], "smoke")
        self.assertEqual(env["REQUIRE_B200"], "0")

    def test_smoke_env_trains_one_prompt_for_one_step_with_two_rollouts(self):
        env = build_smoke_env(
            python_executable="/usr/bin/python3",
            train_file="/data/dapo-train.parquet",
            validation_file="/data/dapo-validation.parquet",
            prepared_model_dir="/tmp/growmtp-prepared",
            run_output_dir="/tmp/growmtp-output",
        )

        self.assertEqual(env["TRAIN_STEPS"], "1")
        self.assertEqual(env["TRAIN_BATCH_SIZE"], "1")
        self.assertEqual(env["ROLLOUT_N"], "2")
        self.assertEqual(env["AGENT_LOOP_WORKERS"], "2")
        self.assertEqual(env["PPO_MINI_BATCH_SIZE"], "1")
        self.assertEqual(env["RESPONSE_LENGTH"], "128")
        self.assertEqual(env["SAVE_FREQ"], "1")
        self.assertEqual(MODAL_GPU, "RTX-PRO-6000")

    def test_b200_smoke_defaults_match_the_valid_two_rollout_batch(self):
        runner = (
            Path(__file__).resolve().parents[1] / "scripts/run_b200_growmtp_lora.sh"
        ).read_text()
        self.assertIn('PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"', runner)
        self.assertIn('AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-$ROLLOUT_N}"', runner)
        self.assertIn('REQUIRE_B200="${REQUIRE_B200:-1}"', runner)

    def test_dependency_constraints_pin_transformers_compatible_kernels(self):
        constraints = (
            Path(__file__).resolve().parents[1]
            / "scripts/modal_growmtp_constraints.txt"
        ).read_text()
        self.assertIn("torch==2.13.0", constraints)
        self.assertIn("flashinfer_python==0.6.18.post1", constraints)
        self.assertIn("apache-tvm-ffi==0.1.11", constraints)
        self.assertIn("sgl-deep-gemm==0.1.7", constraints)
        self.assertIn("sglang-kernel==0.4.7", constraints)
        self.assertIn("nvidia-cutlass-dsl==4.7.1", constraints)
        self.assertIn("kernels==0.14.1", constraints)

    def test_modal_dependencies_skip_local_flashinfer_cubin_bundle(self):
        dependencies = modal_runtime_dependencies(
            [
                "torch==2.13.0",
                "transformers==5.12.1",
                "flashinfer_python==0.6.18.post1",
                "flashinfer_cubin==0.6.18.post1",
            ]
        )

        self.assertEqual(
            dependencies,
            ["flashinfer_python==0.6.18.post1"],
        )

    def test_training_preflight_imports_trainer_after_torchdata(self):
        imported = []

        def successful_import(module_name):
            imported.append(module_name)
            return object()

        verify_training_imports(successful_import)

        self.assertEqual(
            imported,
            ["torchdata.stateful_dataloader", "verl.trainer.main_ppo"],
        )

    def test_training_preflight_names_the_missing_dependency(self):
        def missing_torchdata(module_name):
            if module_name == "torchdata.stateful_dataloader":
                raise ModuleNotFoundError(
                    "No module named 'torchdata'", name="torchdata"
                )
            return object()

        with self.assertRaisesRegex(
            RuntimeError, "torchdata.stateful_dataloader"
        ):
            verify_training_imports(missing_torchdata)


if __name__ == "__main__":
    unittest.main()
