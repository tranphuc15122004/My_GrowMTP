import importlib.util
import ast
from contextlib import nullcontext
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "verl/verl/trainer/mtp/probe.py"


def load_probe():
    assert PROBE_PATH.exists(), "The comparison probe has not been implemented"
    spec = importlib.util.spec_from_file_location("growmtp_probe", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_method(path, class_name, method_name, namespace):
    """Run production methods without importing optional Ray/CUDA transport packages."""
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


def test_exact_shift_and_fixed_draft_lag_have_hand_checked_values():
    probe = load_probe()
    old = torch.tensor([[[0.8, 0.2], [0.8, 0.2]]]).log()
    new = torch.tensor([[[0.2, 0.8], [0.2, 0.8]]]).log()
    metrics, rows = probe.compare_distributions(old, new, old, new)
    assert metrics["draft/probe/kl_new_old_mean"] == pytest.approx(0.6 * math.log(4))
    assert metrics["draft/probe/tv_new_old_mean"] == pytest.approx(0.6)
    assert metrics["draft/probe/tau_pre_surrogate"] == pytest.approx(3)
    assert metrics["draft/probe/tau_post_fixed_draft_surrogate"] == pytest.approx(1.56)
    assert metrics["draft/probe/delta_lag_surrogate"] == pytest.approx(1.44)
    assert metrics["draft/probe/tau_post_joint_surrogate"] == pytest.approx(3)
    assert metrics["draft/probe/joint_update_gain_surrogate"] == pytest.approx(1.44)
    assert rows[0]["alpha_post_fixed_draft"] == pytest.approx([0.4, 0.4])


def test_probe_rows_include_positive_advantage_shift_priority(tmp_path):
    probe = load_probe()
    cases = [{"advantage": 2.0, "trajectory_index": 0},
             {"advantage": -1.0, "trajectory_index": 1}]
    rows = [{"kl_new_old_mean": 0.3}, {"kl_new_old_mean": 0.5}]
    probe.write_probe_rows(tmp_path, 4, cases, rows)
    import json
    logged = [json.loads(line) for line in (tmp_path / "4.jsonl").read_text().splitlines()]
    assert logged[0]["positive_advantage_kl_score"] == pytest.approx(0.6)
    assert logged[1]["positive_advantage_kl_score"] == 0.0


def test_policy_shift_can_improve_acceptance_and_lag_stays_signed():
    probe = load_probe()
    old = torch.tensor([[[0.8, 0.2]]]).log()
    new = torch.tensor([[[0.2, 0.8]]]).log()
    metrics, _ = probe.compare_distributions(old, new, new, new)
    assert metrics["draft/probe/delta_lag_surrogate"] == pytest.approx(-0.6)


def test_identical_distributions_have_zero_shift_and_finite_metrics():
    probe = load_probe()
    logits = torch.tensor([[[80.0, -80.0], [-80.0, 80.0]]])
    logp = logits.log_softmax(-1)
    metrics, _ = probe.compare_distributions(logp, logp, logp, logp)
    assert metrics["draft/probe/kl_new_old_mean"] == 0
    assert metrics["draft/probe/delta_lag_surrogate"] == 0
    assert all(math.isfinite(value) for value in metrics.values())


def test_probe_initializes_fsdp_root_and_restores_actual_reshard_settings():
    probe = load_probe()
    calls = []

    class Sharded(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.name = name
            self.setting = object()
            self.group = SimpleNamespace(post_forward_mesh_info=self.setting)
            self.state = SimpleNamespace(_fsdp_param_group=self.group, _auto_reshard_after_forward=True,
                                         _lazy_init=lambda: calls.append(name + "/init"))

        def _get_fsdp_state(self):
            return self.state

        def set_reshard_after_forward(self, value, **kwargs):
            calls.append(self.name + "/set")
            self.state._auto_reshard_after_forward = False
            self.group.post_forward_mesh_info = None if not value else object()

        def unshard(self):
            calls.append(self.name + "/unshard")

        def reshard(self):
            calls.append(self.name + "/reshard")

    root = Sharded("root")
    root.mtp = Sharded("head")
    with probe.probe_context(root):
        assert root.group.post_forward_mesh_info is None
        assert root.mtp.group.post_forward_mesh_info is None
    assert calls[0] == "root/init"
    assert root.group.post_forward_mesh_info is root.setting
    assert root.mtp.group.post_forward_mesh_info is root.mtp.setting
    assert root.state._auto_reshard_after_forward is True
    assert root.mtp.state._auto_reshard_after_forward is True


def test_probe_selection_keeps_complete_context_and_does_not_mutate_rng():
    probe = load_probe()
    record = {
        "prefix_ids": torch.arange(1, 11),
        "target_ids": torch.arange(11),
        "position": torch.tensor([2, 7]),
        "draft_tokens": torch.tensor([[3, 11, 12], [8, 13, 14]]),
        "accept_len": torch.tensor([1, 2]),
        "trajectory_index": 42,
        "advantage": 0.5,
        "score": 1.0,
    }
    state = torch.random.get_rng_state().clone()
    cases = probe.select_cases([record], max_cycles=4, max_context=6, seed=17)
    assert len(cases) == 1
    assert cases[0]["input_ids"].tolist() == [0, 1, 2, 3, 11]
    assert cases[0]["position"] == 2
    assert cases[0]["trajectory_index"] == 42
    assert torch.equal(state, torch.random.get_rng_state())
    assert record["prefix_ids"].tolist() == list(range(1, 11))


def test_distributed_sample_keeps_global_random_priorities_even_with_unequal_rank_counts(monkeypatch):
    probe = load_probe()

    def record(rank, count):
        return {"prefix_ids": torch.arange(1, count + 3), "target_ids": torch.arange(count + 4),
                "position": torch.arange(count),
                "draft_tokens": torch.tensor([[i + 1, 7, 8] for i in range(count)]),
                "accept_len": torch.ones(count), "trajectory_index": rank, "source_rank": rank,
                "advantage": 0.5, "score": 1.0}

    records = [record(0, 8), record(1, 2)]
    full = [probe.select_cases([item], 10, 20, 17 + i) for i, item in enumerate(records)]
    local = [probe.select_cases([item], 2, 20, 17 + i) for i, item in enumerate(records)]
    expected = sorted(full[0] + full[1], key=lambda case: case["_sampling_priority"], reverse=True)[:2]
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", lambda parts, cases, group: parts.__setitem__(slice(None), local))
    first = probe.synchronized_cases(local[0], 2, group="dp")
    second = probe.synchronized_cases(local[1], 2, group="dp")
    keys = lambda cases: [(case["source_rank"], case["cycle_index"]) for case in cases]
    assert keys(first) == keys(second) == keys(expected)
    # A rank with no cycles still participates and receives the same global cases.
    local[0] = []
    assert keys(probe.synchronized_cases([], 2, group="dp")) == keys(local[1])


def test_tiny_real_target_and_head_probe_preserves_training_state_and_gradients():
    probe = load_probe()
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    head_path = ROOT / "verl/verl/models/transformers/mtp/head.py"
    spec = importlib.util.spec_from_file_location("growmtp_head", head_path)
    head_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(head_module)
    config = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config)
    model.mtp = head_module.MTPHead(config)
    model.train()
    model.mtp.norm.eval()
    case = {"input_ids": torch.tensor([1, 2, 3, 4, 5]), "draft_tokens": torch.tensor([4, 5, 6]),
            "position": 2, "trajectory_index": 0, "advantage": 1.0, "score": 1.0}
    modes = [module.training for module in model.modules()]
    params = [parameter.detach().clone() for parameter in model.parameters()]
    state = torch.random.get_rng_state().clone()
    p, q = probe.evaluate_cases(model, [case], temperature=1.0)
    assert p.shape == q.shape == (1, 2, 16)
    assert torch.allclose(p.exp().sum(-1), torch.ones(1, 2))
    assert torch.allclose(q.exp().sum(-1), torch.ones(1, 2))
    assert modes == [module.training for module in model.modules()]
    assert torch.equal(state, torch.random.get_rng_state())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(old, parameter) for old, parameter in zip(params, model.parameters()))


@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_probe_matches_real_target_and_shifted_mtp_replay(temperature):
    probe = load_probe()
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from transformers.cache_utils import DynamicCache

    spec = importlib.util.spec_from_file_location("growmtp_head_alignment", ROOT / "verl/verl/models/transformers/mtp/head.py")
    head_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(head_module)
    config = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval()
    model.mtp = head_module.MTPHead(config).eval()
    # The first token is absent from prefix_ids: those are MTP's shifted embeddings.
    record = {"target_ids": torch.tensor([1, 2, 3, 4, 7]),
              "prefix_ids": torch.tensor([2, 3, 4]), "position": torch.tensor([2]),
              "draft_tokens": torch.tensor([[4, 5, 6]]), "accept_len": torch.tensor([1]),
              "trajectory_index": 0, "advantage": 1.0, "score": 1.0}
    case = probe.select_cases([record], 1, 8, 0)[0]
    assert case["input_ids"].tolist() == [1, 2, 3, 4, 5]
    p, q = probe.evaluate_cases(model, [case], temperature=temperature)
    with torch.no_grad():
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        hidden = model.model(ids).last_hidden_state
        expected_p = (model.lm_head(hidden[:, 3:5]).float() / temperature).log_softmax(-1)
        # Full prefill produces first draft logits from h(prompt[-1]) + embedding(seed).
        cache = DynamicCache()
        mask = torch.zeros(1, 1, 3, 3).masked_fill(~torch.ones(3, 3, dtype=torch.bool).tril()[None, None], -torch.inf)
        h = model.mtp(hidden[:, :3], model.get_input_embeddings()(ids[:, 1:4]),
                      torch.tensor([[0, 1, 2]]), cache, mask)
        first = model.lm_head(h[:, -1:]).float().log_softmax(-1)
        second_h = model.mtp(h[:, -1:], model.get_input_embeddings()(ids[:, 4:5]),
                            torch.tensor([[3]]), cache, torch.zeros(1, 1, 1, 4))
        second = model.lm_head(second_h).float().log_softmax(-1)
        expected_q = torch.cat([first, second], dim=1)
    torch.testing.assert_close(p, expected_p, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(q, expected_q, atol=1e-6, rtol=1e-6)


def test_validation_schedule_supports_final_only_without_periodic_tests():
    probe = load_probe()
    assert probe.should_validate(3, 3, frequency=-1, final_validation=True)
    assert not probe.should_validate(2, 3, frequency=-1, final_validation=True)
    assert not probe.should_validate(3, 3, frequency=-1, final_validation=False)
    assert probe.should_validate(2, 5, frequency=2, final_validation=False)


def test_engine_snapshot_replays_shifted_nested_records_around_real_optimizer_update(monkeypatch, tmp_path):
    import json
    import sys
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    probe = load_probe()
    monkeypatch.setitem(sys.modules, "verl.trainer.mtp.probe", probe)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    spec = importlib.util.spec_from_file_location("growmtp_engine_head", ROOT / "verl/verl/models/transformers/mtp/head.py")
    head_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(head_module)
    config = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config)
    model.mtp = head_module.MTPHead(config)
    engine = SimpleNamespace(module=model, ulysses_sequence_parallel_size=1,
                             model_config=SimpleNamespace(mtp=SimpleNamespace(
                                 comparison_probe_max_cycles=4, comparison_probe_max_context=8,
                                 comparison_refresh_fraction=0.0)),
                             engine_config=SimpleNamespace(reshard_after_forward=True),
                             get_data_parallel_rank=lambda: 0, get_data_parallel_group=lambda: None)
    namespace = {"torch": torch, "math": math, "FSDP": type("FSDP1", (), {}),
                 "tu": SimpleNamespace(get=lambda data, key: data[key])}
    path = ROOT / "verl/verl/workers/engine/fsdp/transformer_impl.py"
    capture = load_method(path, "FSDPEngine", "capture_comparison_probe", namespace)
    finish = load_method(path, "FSDPEngine", "finish_comparison_probe", namespace)
    nested = lambda values: torch.nested.nested_tensor(values, layout=torch.jagged)
    data = {"mtp_num_cycles": torch.tensor([1]), "input_ids": nested([torch.tensor([1, 2, 3, 4, 7])]),
            "mtp_prefix_ids": nested([torch.tensor([2, 3, 4])]),
            "mtp_position": nested([torch.tensor([2])]),
            "mtp_draft_tokens": nested([torch.tensor([[4, 5, 6]])]),
            "mtp_accept_len": nested([torch.tensor([1])]), "temperature": 1.0,
            "comparison_advantage": torch.tensor([0.5]), "comparison_score": torch.tensor([1.0]),
            "comparison_request_index": torch.tensor([37])}
    snapshot = capture(engine, data, 4)
    frozen_draft = snapshot["fixed_draft"].clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.tensor([[1, 2, 3, 4, 7]]), labels=torch.tensor([[1, 2, 3, 4, 7]])).loss
    loss = loss + model.mtp.fc.weight.square().mean()
    loss.backward()
    optimizer.step()
    params = [value.detach().clone() for value in model.parameters()]
    grads = [None if value.grad is None else value.grad.clone() for value in model.parameters()]
    metrics = finish(engine, snapshot, str(tmp_path))
    assert metrics["draft/probe/num_cycles"] == 1
    assert metrics["draft/probe/kl_new_old_mean"] > 0
    assert torch.equal(frozen_draft, snapshot["fixed_draft"])
    assert all(torch.equal(old, value) for old, value in zip(params, model.parameters()))
    assert all(value.grad is None if old is None else torch.equal(old, value.grad)
               for old, value in zip(grads, model.parameters()))
    row = json.loads((tmp_path / "4.jsonl").read_text())
    assert row["trajectory_index"] == 37
    assert row["positive_advantage_kl_score"] == pytest.approx(0.5 * row["kl_new_old_mean"])
    empty = capture(engine, {"mtp_num_cycles": torch.tensor([0]), "input_ids": data["input_ids"]}, 8)
    empty_metrics = finish(engine, empty)
    assert empty_metrics["draft/probe/num_cycles"] == 0
    assert empty_metrics["draft/probe/eligible_cycles"] == 0
    assert "draft/probe/kl_new_old_mean" not in empty_metrics


def test_per_trajectory_export_uses_masked_advantage_and_real_verification_counts(monkeypatch, tmp_path):
    import json
    import sys
    import numpy as np

    probe = load_probe()
    spec = importlib.util.spec_from_file_location("growmtp_export_metrics", ROOT / "verl/verl/trainer/mtp/metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    monkeypatch.setitem(sys.modules, "verl.trainer.mtp.probe", probe)
    monkeypatch.setitem(sys.modules, "verl.trainer.mtp.metrics", metrics)
    method = load_method(ROOT / "verl/verl/trainer/ppo/ray_trainer.py", "RayPPOTrainer",
                         "_log_comparison_trajectories", {"np": np})
    trainer = SimpleNamespace(global_steps=4, _comparison_root=lambda: tmp_path)
    batch = SimpleNamespace(batch={"advantages": torch.tensor([[1., 1., 999.], [0., 0., 0.]]),
                                   "response_mask": torch.tensor([[1, 1, 0], [0, 0, 0]]),
                                   "token_level_scores": torch.tensor([[0., 1., 0.], [float("nan"), 0., 0.]]),
                                   "token_level_rewards": torch.tensor([[0., 0.5, 0.], [0., 0., 0.]])},
                            non_tensor_batch={"uid": ["shared-prompt", "shared-prompt"],
                                              "spec_num_draft_tokens": [6, 0],
                                              "spec_num_accepted_tokens": [3, 0],
                                              "spec_num_verify_steps": [2, 0]})
    method(trainer, batch)
    rows = [json.loads(line) for line in (tmp_path / "trajectories/4.jsonl").read_text().splitlines()]
    assert rows[0]["advantage"] == 1
    assert rows[0]["response_tokens"] == 2
    assert rows[0]["acceptance_length"] == 2.5
    assert rows[0]["acceptance_rate"] == 0.5
    assert rows[0]["prompt_group_id"] == rows[1]["prompt_group_id"]
    assert rows[1]["advantage"] is None and rows[1]["score"] is None
    assert rows[1]["acceptance_length"] is None


@pytest.mark.parametrize("refresh, fail_train", [(False, False), (True, False), (True, True)])
def test_actor_probe_brackets_all_minibatches_instead_of_individual_updates(refresh, fail_train):
    # Execute the real orchestration method; Ray transport/GPU engine are external here.
    tree = ast.parse((ROOT / "verl/verl/workers/engine_workers.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrainingWorker")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "train_mini_batch")
    method.decorator_list = []
    calls = []
    import tempfile
    cache = tempfile.TemporaryFile()

    class Data(dict):
        shape = (2,)

        def cpu(self):
            return self

    tu = SimpleNamespace(
        pop=lambda data, key, default=None: data.pop(key, default),
        make_iterator=lambda *args, **kwargs: [Data(), Data()],
        assign_non_tensor=lambda data, **kwargs: data.update(kwargs),
        get=lambda data, key: data[key],
        get_tensordict=lambda tensor_dict, non_tensor_dict: Data(non_tensor_dict),
    )

    def append_to_dict(target, values):
        for key, value in values.items():
            target.setdefault(key, []).append(value)

    def capture(data, step):
        calls.append("capture")
        return {"step": step, "defer_head_update": refresh, "shift": {"cache": cache}}

    def finish(snapshot, output_dir):
        assert snapshot["step"] == 4
        calls.append("finish")
        return {"draft/probe/kl_new_old_mean": 0.25}

    def train(data):
        assert data["mtp_defer_head_update"] == refresh
        calls.append("train")
        if fail_train:
            raise RuntimeError("actor update failed")
        return {"metrics": {"loss": 1.0}}

    engine = SimpleNamespace(
        train_mode=lambda **kwargs: nullcontext(), get_data_parallel_size=lambda: 1,
        get_data_parallel_rank=lambda: 0, is_mp_src_rank_with_outputs=lambda: True,
        capture_comparison_probe=capture, finish_comparison_probe=finish,
        _mtp_update_seconds=0.0, get_data_parallel_group=lambda: None,
    )
    worker = SimpleNamespace(engine=engine, train_batch=train, device_name="cpu",
                             model_config=SimpleNamespace(mtp=SimpleNamespace(growmtp=True)))
    import time
    namespace = {"tu": tu, "maybe_fix_3d_position_ids": lambda data: None,
                 "Timer": lambda **kwargs: nullcontext(), "NonTensorData": lambda value: value,
                 "torch": torch, "chain": __import__("itertools").chain,
                 "Metric": type("UnusedMetric", (), {}), "append_to_dict": append_to_dict,
                 "time": time}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "actor_probe_orchestration", "exec"), namespace)
    data = Data(mini_batch_size=1, epochs=1, comparison_probe_step=4,
                comparison_probe_output_dir="unused")
    if fail_train:
        with pytest.raises(RuntimeError, match="actor update failed"):
            namespace["train_mini_batch"](worker, data)
        assert cache.closed
        assert calls == ["capture", "train"]
        return
    output = namespace["train_mini_batch"](worker, data)
    assert cache.closed
    assert calls == ["capture", "train", "train", "finish"]
    assert output["metrics"]["draft/probe/kl_new_old_mean"] == [0.25]
    assert output["metrics"]["draft/probe/time_s"][0] >= 0
