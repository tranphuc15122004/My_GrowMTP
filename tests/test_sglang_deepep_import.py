"""A dense SGLang rollout must not initialize optional DeepEP at import time."""

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest


DEEPEP_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py"
)


def test_optional_deepep_is_loaded_only_when_requested():
    class DeepEPMode:
        AUTO = object()
        NORMAL = object()
        LOW_LATENCY = object()

    base_classes = dict.fromkeys(
        (
            "BaseDispatcher",
            "BaseDispatcherConfig",
            "DispatcherBaseHooks",
        ),
        object,
    )
    fake_modules = {
        "sglang.srt.distributed.parallel_state": SimpleNamespace(get_tp_group=None),
        "sglang.srt.environ": SimpleNamespace(envs=None),
        "sglang.srt.eplb.expert_distribution": SimpleNamespace(
            get_global_expert_distribution_recorder=None
        ),
        "sglang.srt.layers": SimpleNamespace(deep_gemm_wrapper=None),
        "sglang.srt.layers.dp_attention": SimpleNamespace(get_is_extend_in_batch=None),
        "sglang.srt.layers.moe.token_dispatcher.base": SimpleNamespace(
            **base_classes,
            CombineInput=type,
            CombineInputFormat=SimpleNamespace(),
            DispatchOutput=type,
            DispatchOutputFormat=SimpleNamespace(),
        ),
        "sglang.srt.layers.moe.topk": SimpleNamespace(TopKOutput=object),
        "sglang.srt.layers.moe.utils": SimpleNamespace(
            DeepEPMode=DeepEPMode,
            get_deepep_config=lambda: None,
            get_moe_runner_backend=None,
            is_tbo_enabled=None,
        ),
        "sglang.srt.utils": SimpleNamespace(
            get_bool_env_var=lambda _: False,
            is_blackwell=None,
            is_hip=lambda: False,
            is_npu=lambda: False,
            load_json_config=None,
        ),
        "sglang.srt.layers.quantization.fp8_kernel": SimpleNamespace(
            sglang_per_token_group_quant_fp8=None
        ),
    }
    real_import = builtins.__import__

    def isolated_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "deep_ep":
            raise AssertionError("Duplicate NCCL runtime found")
        if name in fake_modules:
            return fake_modules[name]
        return real_import(name, globals, locals, fromlist, level)

    namespace = {
        "__name__": "sglang.srt.layers.moe.token_dispatcher.deepep",
        "__builtins__": {**vars(builtins), "__import__": isolated_import},
    }
    exec(compile(DEEPEP_SOURCE.read_text(), str(DEEPEP_SOURCE), "exec"), namespace)

    with pytest.raises(AssertionError, match="Duplicate NCCL runtime found"):
        namespace["DeepEPConfig"]()
