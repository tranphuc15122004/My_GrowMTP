"""Review checks against production methods, without Ray/CUDA transport imports."""
import ast
import copy
import importlib
import math
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace, MethodType

import pytest
import torch
from torch.distributed.tensor import DTensor
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'verl/verl/workers/engine/fsdp/transformer_impl.py'


def method(name, namespace):
    tree = ast.parse(SOURCE.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'FSDPEngine')
    node = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)
    return namespace[name]


@pytest.fixture
def engine(monkeypatch):
    # Only skip package initializers that require Ray. All math/model/replay code is real.
    for name, folder in (
        ('verl', 'verl/verl'), ('verl.trainer', 'verl/verl/trainer'),
        ('verl.trainer.mtp', 'verl/verl/trainer/mtp'),
        ('verl.models', 'verl/verl/models'),
        ('verl.models.transformers', 'verl/verl/models/transformers'),
    ):
        package = ModuleType(name)
        package.__path__ = [str(ROOT / folder)]
        monkeypatch.setitem(sys.modules, name, package)
    mtp = importlib.import_module('verl.models.transformers.mtp')
    replay_engine = importlib.import_module('verl.trainer.mtp.engine')
    probe = importlib.import_module('verl.trainer.mtp.probe')
    signals = importlib.import_module('verl.trainer.mtp.signals')
    torch.manual_seed(20)
    text = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=8)
    text._attn_implementation = 'sdpa'
    model = Qwen3ForCausalLM(text)
    model.mtp = mtp.MTPHead(text)
    config = SimpleNamespace(growmtp=True, comparison_probe_max_cycles=4,
                             comparison_probe_max_context=16, comparison_refresh_fraction=.25,
                             speculative_num_steps=2, teacher_topk=8, chunk_size=2,
                             vocab_chunk_size=2, mtp_loss_scaling_factor=.1,
                             learning_rate=.02)
    optimizer = torch.optim.AdamW(replay_engine.optimizer_groups(model, config),
                                  lr=.01, weight_decay=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1/(step+1))

    def group_norm(parameters):
        grads = [p.grad.flatten() for p in parameters if p.grad is not None]
        return torch.cat(grads).norm() if grads else torch.zeros(())

    namespace = dict(torch=torch, math=math, time=time, DTensor=DTensor,
                     FSDP=type('FSDP1', (), {}), FSDPModule=type('FSDP2', (), {}),
                     fsdp2_grad_norm=group_norm,
                     fsdp2_clip_grad_norm_=torch.nn.utils.clip_grad_norm_,
                     tu=SimpleNamespace(get=lambda data, key: data[key]))
    result = SimpleNamespace(module=model, optimizer=optimizer, lr_scheduler=scheduler,
                             model_config=SimpleNamespace(mtp=config),
                             optimizer_config=SimpleNamespace(clip_grad=1.),
                             ulysses_sequence_parallel_size=1,
                             get_data_parallel_size=lambda: 1,
                             get_data_parallel_rank=lambda: 0,
                             get_data_parallel_group=lambda: None,
                             scaler=None, _qat_enabled=False)
    for name in ('capture_comparison_probe', 'finish_comparison_probe',
                 'refresh_draft_teacher', 'optimizer_step'):
        setattr(result, name, MethodType(method(name, namespace), result))
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    result.probe = probe
    result.replay_engine = replay_engine
    result.signals = signals
    return result


def rollout_data(engine, counts=(1,)):
    nested = lambda values: torch.nested.nested_tensor(values, layout=torch.jagged)
    rows = {key: [] for key in ('input_ids', 'prefix_ids', 'prefix_hidden', 'position',
                               'draft_tokens', 'accept_len', 'hidden', 'topk_val', 'topk_idx')}
    for count in counts:
        ids = torch.arange(1, 2 * count + 4)
        positions = torch.arange(2, 2 * count + 1, 2)
        tokens = torch.tensor([[int(ids[p+1]), 7, 8] for p in positions])
        cases = [dict(input_ids=torch.cat([ids[:int(p)+1], t[:-1]]),
                      position=int(p), draft_tokens=t) for p, t in zip(positions, tokens)]
        with torch.no_grad():
            h = engine.module.model(ids[None]).last_hidden_state[0]
        target, _ = engine.probe.evaluate_cases(engine.module, cases)
        values, indices = target.topk(8, dim=-1)
        row = dict(input_ids=ids, prefix_ids=ids[1:int(positions[-1])+2],
                   prefix_hidden=h[:int(positions[-1])+1], position=positions,
                   draft_tokens=tokens, accept_len=torch.ones(count, dtype=torch.long),
                   hidden=h[positions], topk_val=values.bfloat16(), topk_idx=indices)
        for key, value in row.items():
            rows[key].append(value)
    return dict(mtp_num_cycles=torch.tensor(counts), input_ids=nested(rows.pop('input_ids')),
                **{'mtp_'+key: nested(values) for key, values in rows.items()}, temperature=1.,
                comparison_advantage=torch.full((len(counts),), .5),
                comparison_score=torch.ones(len(counts)),
                comparison_request_index=torch.arange(37, 37+len(counts)))


def rl_update(engine):
    ids = torch.tensor([[1, 2, 3, 4, 7]])
    engine.module(ids, labels=ids).loss.backward()
    engine.optimizer.step()
    engine.optimizer.zero_grad(set_to_none=True)
    engine.lr_scheduler.step()


def test_real_refresh_uses_updated_teacher_and_changes_only_head(engine, monkeypatch, tmp_path):
    snapshot = engine.capture_comparison_probe(rollout_data(engine), 4)
    rl_update(engine)
    target_before = {n: p.detach().clone() for n, p in engine.module.named_parameters()
                     if not n.startswith('mtp.')}
    head_before = [p.detach().clone() for p in engine.module.mtp.parameters()]
    lrs = [g['lr'] for g in engine.optimizer.param_groups]
    scheduler = copy.deepcopy(engine.lr_scheduler.state_dict())
    target_steps = [engine.optimizer.state[p]['step'].clone()
                    for p in engine.optimizer.param_groups[0]['params']]
    new_p, _, hidden = engine.probe.evaluate_cases(engine.module, snapshot['cases'], return_hidden=True)
    assert not torch.equal(new_p, snapshot['old_target'])
    expected_values, expected_ids = new_p[0].topk(8, dim=-1)
    observed = []
    original = engine.replay_engine.backward_head

    def inspect_teacher(model, data, config, global_cycles, dp_size, **kwargs):
        observed.extend(engine.signals.unpack_signals(data))
        return original(model, data, config, global_cycles, dp_size, **kwargs)

    monkeypatch.setattr(engine.replay_engine, 'backward_head', inspect_teacher)
    metrics = engine.finish_comparison_probe(snapshot, tmp_path)
    assert metrics['draft/refresh/selected_count'] == 1
    assert metrics['draft/refresh/applied'] == 1
    assert any(not torch.equal(p, old) for p, old in zip(engine.module.mtp.parameters(), head_before))
    assert all(torch.equal(p, target_before[n]) for n, p in engine.module.named_parameters()
               if n in target_before)
    assert all(torch.equal(engine.optimizer.state[p]['step'], step)
               for p, step in zip(engine.optimizer.param_groups[0]['params'], target_steps))
    assert all(engine.optimizer.state[p]['step'] == 1 for p in engine.module.mtp.parameters())
    assert all(p.grad is None for p in engine.module.parameters())
    assert engine._optimizer_step_metrics['target/optimizer/step_applied_fraction'] == 0
    assert engine._optimizer_step_metrics['draft/optimizer/step_applied_fraction'] == 1
    assert [g['lr'] for g in engine.optimizer.param_groups] == lrs
    assert engine.lr_scheduler.state_dict() == scheduler
    assert torch.equal(observed[0]['topk_val'][0], expected_values.bfloat16())
    assert torch.equal(observed[0]['topk_idx'][0], expected_ids)
    assert torch.equal(observed[0]['hidden'][0], hidden[0][2])
    assert observed[0]['prefix_ids'].tolist() == [2, 3, 4]
    assert observed[0]['accept_len'].tolist() == [1]


@pytest.mark.parametrize('advantage,fraction', [(-.5, .25), (.5, 0), (0, .25)])
def test_refresh_skips_zero_score_or_zero_fraction(engine, advantage, fraction):
    data = rollout_data(engine)
    data['comparison_advantage'].fill_(advantage)
    engine.model_config.mtp.comparison_refresh_fraction = fraction
    snapshot = engine.capture_comparison_probe(data, 4)
    rl_update(engine)
    before = [p.detach().clone() for p in engine.module.parameters()]
    metrics = engine.finish_comparison_probe(snapshot)
    assert metrics['draft/refresh/selected_count'] == 0
    assert metrics.get('draft/refresh/applied', 0) == 0
    # Deferred steps still train unselected trajectories once against their old teacher.
    if fraction == 0:
        assert all(torch.equal(p, old) for p, old in zip(engine.module.parameters(), before))
    else:
        assert any(not torch.equal(p, old) for (name, p), old in zip(engine.module.named_parameters(), before)
                   if name.startswith('mtp.'))


def test_nonfinite_skipped_optimizer_step_is_not_reported_applied(engine, monkeypatch):
    snapshot = engine.capture_comparison_probe(rollout_data(engine), 4)
    rl_update(engine)
    before = [p.detach().clone() for p in engine.module.parameters()]

    def bad_gradient(model, data, config, global_cycles, dp_size, **kwargs):
        next(model.mtp.parameters()).grad = torch.full_like(next(model.mtp.parameters()), float('inf'))
        return {'mtp/dca_loss': float('inf')}

    monkeypatch.setattr(engine.replay_engine, 'backward_head', bad_gradient)
    metrics = engine.finish_comparison_probe(snapshot)
    assert all(torch.equal(p, old) for p, old in zip(engine.module.parameters(), before))
    assert engine._optimizer_step_metrics['draft/optimizer/step_applied_fraction'] == 0
    assert metrics['draft/refresh/applied'] == 0
    assert metrics['draft/refresh/updated_cycles'] == 0


def test_refresh_teacher_matches_growmtp_raw_logits_when_temperature_is_not_one(engine, monkeypatch):
    data = rollout_data(engine)
    data['temperature'] = .6
    snapshot = engine.capture_comparison_probe(data, 4)
    rl_update(engine)
    with torch.no_grad():
        case = snapshot['cases'][0]
        hidden = engine.module.model(case['input_ids'][None]).last_hidden_state[0]
        # SGLang records raw teacher logits independently of rollout temperature.
        raw_teacher = engine.module.lm_head(hidden[case['position']+1:]).float().log_softmax(-1)
        expected_values, expected_ids = raw_teacher.topk(8, dim=-1)
    observed = []
    original = engine.replay_engine.backward_head

    def inspect_teacher(model, batch, config, global_cycles, dp_size, **kwargs):
        observed.extend(engine.signals.unpack_signals(batch))
        return original(model, batch, config, global_cycles, dp_size, **kwargs)

    monkeypatch.setattr(engine.replay_engine, 'backward_head', inspect_teacher)
    engine.finish_comparison_probe(snapshot)
    assert torch.equal(observed[0]['topk_idx'][0], expected_ids)
    torch.testing.assert_close(observed[0]['topk_val'][0], expected_values.bfloat16(), rtol=0, atol=0)


def test_top_fraction_ranks_all_trajectories_and_replaces_every_selected_cycle(engine, monkeypatch):
    data = rollout_data(engine, counts=(2,)*8)
    old = [row.clone() for row in data['mtp_topk_val'].unbind()]
    snapshot = engine.capture_comparison_probe(data, 4)
    rl_update(engine)
    observed = []
    original = engine.replay_engine.backward_head

    def inspect(model, batch, config, global_cycles, dp_size, **kwargs):
        observed.extend(engine.signals.unpack_signals(batch))
        assert global_cycles == 16
        assert kwargs['global_trajectories'] == 8
        return original(model, batch, config, global_cycles, dp_size, **kwargs)

    monkeypatch.setattr(engine.replay_engine, 'backward_head', inspect)
    metrics = engine.finish_comparison_probe(snapshot)
    assert metrics['draft/refresh/candidate_count'] == 8
    assert metrics['draft/refresh/selected_count'] == 2
    assert metrics['draft/refresh/updated_cycles'] == 4
    assert len(observed) == 8
    changed = [i for i, row in enumerate(observed) if not torch.equal(row['topk_val'], old[i])]
    assert len(changed) == 2
    assert all(observed[i]['topk_val'].shape[0] == 2 for i in changed)
    assert all(engine.optimizer.state[p]['step'] == 1 for p in engine.module.mtp.parameters())


def test_policy_backward_defers_head_on_refresh_steps(engine, monkeypatch):
    from contextlib import nullcontext
    data = rollout_data(engine)
    data.update(loss_mask=torch.ones(1), mtp_defer_head_update=True)
    calls = []
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda *args, **kwargs: None)
    monkeypatch.setattr(engine.replay_engine, 'backward_head', lambda *args, **kwargs: calls.append('head') or
                        {'mtp/dca_loss': 0.})
    namespace = dict(torch=torch, time=time, nullcontext=nullcontext,
                     get_device_id=lambda: 'cpu',
                     tu=SimpleNamespace(assign_non_tensor=lambda data, **kw: data.update(kw),
                                        get=lambda data, key, default=None: data.get(key, default)),
                     prepare_micro_batches=lambda **kw: ([kw['data']], []),
                     postprocess_batch_func=lambda **kw: kw['output_lst'])
    forward = method('forward_backward_batch', namespace)
    parameter = next(engine.module.model.parameters())
    engine.forward_step = lambda *args, **kwargs: (parameter.square().mean(), {'metrics': {}})
    forward(engine, data, None)
    assert calls == []
    assert all(p.grad is None for p in engine.module.mtp.parameters())
    engine.optimizer.zero_grad()
    data['mtp_defer_head_update'] = False
    forward(engine, data, None)
    assert calls == ['head']


def test_deferred_loss_weights_trajectories_equally_with_unequal_cycle_counts(engine, monkeypatch):
    data = rollout_data(engine, counts=(1,3))
    parameter = next(engine.module.mtp.parameters())
    engine.model_config.mtp.mtp_loss_scaling_factor = 1.

    def losses(head, embed, lm, record, *args):
        n = record['position'].numel()
        coefficient = 1. if n == 1 else 10.
        yield parameter.flatten()[0].expand(n) * coefficient, torch.ones(n, 2)

    monkeypatch.setattr(engine.replay_engine, 'replay_chunks', losses)
    engine.replay_engine.backward_head(engine.module, data, engine.model_config.mtp, 4, 1,
                                       global_trajectories=2)
    assert parameter.grad.flatten()[0].item() == pytest.approx(5.5)


def test_diagnostic_context_budget_does_not_disable_algorithm(engine):
    engine.model_config.mtp.comparison_probe_max_context = 1
    snapshot = engine.capture_comparison_probe(rollout_data(engine, counts=(2,)*8), 4)
    assert snapshot['cases'] == []
    rl_update(engine)
    metrics = engine.finish_comparison_probe(snapshot)
    assert metrics['draft/probe/num_cycles'] == 0
    assert metrics['draft/refresh/selected_count'] == 2
    assert metrics['draft/refresh/updated_cycles'] == 4
    assert snapshot['shift']['cache'].closed


def test_empty_verification_batch_does_not_step_head_optimizer(engine):
    nested = lambda rows: torch.nested.nested_tensor(rows, layout=torch.jagged)
    data = dict(mtp_num_cycles=torch.zeros(3, dtype=torch.long),
                input_ids=nested([torch.tensor([1, 2])] * 3))
    snapshot = engine.capture_comparison_probe(data, 4)
    before = [p.detach().clone() for p in engine.module.parameters()]
    metrics = engine.finish_comparison_probe(snapshot)
    assert metrics['draft/refresh/head_step_applied'] == 0
    assert metrics['draft/refresh/selected_count'] == metrics['draft/refresh/updated_cycles'] == 0
    assert engine.optimizer.state == {}
    assert all(torch.equal(p, old) for p, old in zip(engine.module.parameters(), before))


def test_unchanged_target_selects_nothing_and_trains_old_teacher_once(engine):
    snapshot = engine.capture_comparison_probe(rollout_data(engine), 4)
    metrics = engine.finish_comparison_probe(snapshot)
    assert metrics['draft/refresh/selected_count'] == metrics['draft/refresh/applied'] == 0
    assert metrics['draft/shift/kl_new_old_mean'] == 0
    assert metrics['draft/refresh/head_step_applied'] == 1
    assert all(engine.optimizer.state[p]['step'] == 1 for p in engine.module.mtp.parameters())


def test_shift_score_averages_all_cycles_and_uses_positive_advantage(engine, tmp_path):
    import json
    data = rollout_data(engine, counts=(1,3,2,1))
    data['comparison_advantage'] = torch.tensor([.1, 1., -.5, 0.])
    snapshot = engine.capture_comparison_probe(data, 4)
    from verl.trainer.mtp.refresh import record_cases
    positive = snapshot['shift']['positive']
    old, _ = engine.probe.evaluate_cases(engine.module, list(record_cases(positive)))
    rl_update(engine)
    new, _ = engine.probe.evaluate_cases(engine.module, list(record_cases(positive)))
    kl = (new.exp() * (new-old)).sum(-1).clamp_min(0).mean(-1)
    expected = [kl[0].item(), kl[1:].mean().item(), 0., 0.]
    metrics = engine.finish_comparison_probe(snapshot, tmp_path / 'probes')
    assert metrics['draft/refresh/selected_count'] == 1
    rows = [json.loads(line) for line in (tmp_path / 'shifts/4.jsonl').read_text().splitlines()]
    assert [row['trajectory_index'] for row in rows] == [37,38,39,40]
    assert [row['kl_new_old_mean'] for row in rows[:2]] == pytest.approx(expected[:2])
    assert all(row['kl_new_old_mean'] is None for row in rows[2:])
    assert [row['positive_advantage_kl_score'] for row in rows] == pytest.approx(
        [expected[0]*.1, expected[1], 0., 0.])
    assert rows[1]['refresh_selected']
    assert all(not row['refresh_selected'] for row in (rows[0],rows[2],rows[3]))
    assert all(row['kl_measured'] == (i < 2) for i,row in enumerate(rows))


def test_aux_ce_updates_only_positive_advantage_trajectory_head(engine):
    data = rollout_data(engine, counts=(1, 1))
    nested = lambda rows: torch.nested.nested_tensor(rows, layout=torch.jagged)
    data['mtp_target_tokens'] = nested([torch.tensor([[5, 7]]), torch.tensor([[6, 8]])])
    data['mtp_target_mask'] = nested([torch.ones(1, 2, dtype=torch.bool)] * 2)
    data['response_mask'] = torch.tensor([[1, 1], [1, 1]])
    data['advantages'] = torch.tensor([[3., 3.], [-1., -1.]])

    def head_gradient():
        return torch.cat([p.grad.flatten() for p in engine.module.mtp.parameters()
                          if p.grad is not None])

    config = engine.model_config.mtp
    config.rollout_aux_ce_lambda = 0.
    engine.module.zero_grad(set_to_none=True)
    engine.replay_engine.backward_head(engine.module, data, config, 2, 1)
    baseline_gradient = head_gradient().clone()

    config.rollout_aux_ce_lambda = .05
    config.rollout_aux_advantage_clip = 2.
    engine.module.zero_grad(set_to_none=True)
    metrics = engine.replay_engine.backward_head(engine.module, data, config, 2, 1)
    assert metrics['mtp/aux_ce_valid_cycles'] == 2
    assert metrics['mtp/aux_ce_positive_cycles'] == 1
    assert metrics['mtp/aux_ce_loss'] > 0
    assert metrics['mtp/aux_ce_contribution'] == pytest.approx(.05 * metrics['mtp/aux_ce_loss'])
    assert not torch.allclose(head_gradient(), baseline_gradient)
    assert all(p.grad is None for name, p in engine.module.named_parameters()
               if not name.startswith('mtp.'))

    data['advantages'].fill_(-1)
    engine.module.zero_grad(set_to_none=True)
    no_positive = engine.replay_engine.backward_head(engine.module, data, config, 2, 1)
    assert no_positive['mtp/aux_ce_positive_cycles'] == 0
    assert no_positive['mtp/aux_ce_contribution'] == 0
    torch.testing.assert_close(head_gradient(), baseline_gradient)
