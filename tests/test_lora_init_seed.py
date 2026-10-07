import ast
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "verl" / "verl" / "workers" / "engine" / "fsdp" / "transformer_impl.py"


def lora_method(namespace):
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FSDPEngine")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_build_lora_module")
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["_build_lora_module"]


class _Model:
    def enable_input_require_grads(self):
        pass


def test_lora_adapter_initialization_uses_configured_seed(monkeypatch):
    observed = []

    def capture_lora_init(model, config):
        observed.append(torch.rand(()).item())
        return model

    build_lora = lora_method({
        "torch": torch,
        "get_peft_model": capture_lora_init,
        "LoraConfig": lambda **kwargs: kwargs,
        "TaskType": SimpleNamespace(CAUSAL_LM="causal_lm"),
        "convert_to_regular_types": lambda value: value,
    })
    engine = SimpleNamespace(model_config=SimpleNamespace(
        lora_adapter_path=None,
        lora_init_seed=23,
        lora_rank=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        target_parameters=None,
        exclude_modules=None,
    ))

    build_lora(engine, _Model())
    build_lora(engine, _Model())

    assert len(observed) == 2
    assert observed[0] == observed[1]


def test_lora_adapter_without_seed_keeps_existing_rng_behavior(monkeypatch):
    observed = []

    def capture_lora_init(model, config):
        observed.append(torch.rand(()).item())
        return model

    build_lora = lora_method({
        "torch": torch,
        "get_peft_model": capture_lora_init,
        "LoraConfig": lambda **kwargs: kwargs,
        "TaskType": SimpleNamespace(CAUSAL_LM="causal_lm"),
        "convert_to_regular_types": lambda value: value,
    })
    engine = SimpleNamespace(model_config=SimpleNamespace(
        lora_adapter_path=None,
        lora_init_seed=None,
        lora_rank=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        target_parameters=None,
        exclude_modules=None,
    ))
    torch.manual_seed(101)

    build_lora(engine, _Model())
    build_lora(engine, _Model())

    assert len(observed) == 2
    assert observed[0] != observed[1]
