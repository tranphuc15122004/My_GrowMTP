"""Compatibility checks for CUDA SGLang kernels that vary by wheel version."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType

import pytest


GPTQ_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "sglang/python/sglang/srt/layers/quantization/gptq.py"
)


def _load_cuda_import_branch(monkeypatch, exports: dict[str, object]):
    """Execute the real CUDA import branch with a controlled sgl_kernel API."""
    kernel_module = ModuleType("sgl_kernel")
    for name, value in exports.items():
        setattr(kernel_module, name, value)
    monkeypatch.setitem(sys.modules, "sgl_kernel", kernel_module)

    sglang_module = ModuleType("sglang")
    sglang_module.__path__ = []
    jit_module = ModuleType("sglang.jit_kernel")
    jit_module.__path__ = []
    repack_module = ModuleType("sglang.jit_kernel.gptq_marlin_repack")
    repack_module.gptq_marlin_repack = object()
    sglang_module.jit_kernel = jit_module
    jit_module.gptq_marlin_repack = repack_module
    monkeypatch.setitem(sys.modules, "sglang", sglang_module)
    monkeypatch.setitem(sys.modules, "sglang.jit_kernel", jit_module)
    monkeypatch.setitem(
        sys.modules, "sglang.jit_kernel.gptq_marlin_repack", repack_module
    )

    tree = ast.parse(GPTQ_SOURCE.read_text())
    branch = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "_is_cuda"
    )
    namespace = {"_is_cuda": True}
    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(GPTQ_SOURCE), "exec"), namespace)
    return namespace


def _load_gptq_guard():
    tree = ast.parse(GPTQ_SOURCE.read_text())
    helper = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_require_gptq_kernel"
        ),
        None,
    )
    assert helper is not None, "GPTQ calls need a guard when optional kernels are absent"
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(GPTQ_SOURCE), "exec"), namespace)
    return namespace["_require_gptq_kernel"]


def test_missing_legacy_gptq_ops_do_not_block_cuda_module_import(monkeypatch):
    namespace = _load_cuda_import_branch(monkeypatch, {})
    assert namespace["gptq_gemm"] is None
    assert namespace["gptq_shuffle"] is None


def test_available_gptq_ops_are_preserved(monkeypatch):
    gemm = object()
    shuffle = object()
    namespace = _load_cuda_import_branch(
        monkeypatch, {"gptq_gemm": gemm, "gptq_shuffle": shuffle}
    )
    assert namespace["gptq_gemm"] is gemm
    assert namespace["gptq_shuffle"] is shuffle


def test_using_gptq_without_legacy_kernel_fails_with_actionable_error():
    require_kernel = _load_gptq_guard()
    with pytest.raises(RuntimeError, match="sglang-kernel.*GPTQ"):
        require_kernel("gptq_gemm", None)


def test_gptq_guard_returns_available_kernel_unchanged():
    require_kernel = _load_gptq_guard()
    kernel = object()
    assert require_kernel("gptq_gemm", kernel) is kernel
