import unittest

from scripts.check_training_imports import verify_training_imports
from scripts.run_modal_growmtp_smoke import build_smoke_env


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
