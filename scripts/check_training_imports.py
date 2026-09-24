#!/usr/bin/env python3
"""Fail early when Python dependencies for the actual trainer are missing."""

from __future__ import annotations

import importlib
from collections.abc import Callable

TRAINING_IMPORTS = ("torchdata.stateful_dataloader", "verl.trainer.main_ppo")


def verify_training_imports(import_module: Callable[[str], object] = importlib.import_module) -> None:
    for module_name in TRAINING_IMPORTS:
        try:
            import_module(module_name)
        except Exception as exc:
            raise RuntimeError(
                f"GrowMTP training preflight failed importing {module_name}: "
                f"{type(exc).__name__}: {exc}. Install the missing dependency in the selected venv, "
                "then rerun this preflight."
            ) from exc
        print(f"Training import OK: {module_name}")


if __name__ == "__main__":
    verify_training_imports()
