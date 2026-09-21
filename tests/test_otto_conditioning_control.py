"""Synthetic conditioning-control contracts; no saved models or simulator."""
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('conditioning_control_test', ROOT/'scripts/study_otto_conditioning_control.py')
S = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = S
SPEC.loader.exec_module(S)
ARMS = tuple(f'{kind}@{seed}' for seed in (10101, 10102, 10103) for kind in ('gain1', 'gain53')) + ('analytic_inbounds',)
FIRST = {'lambda3': 15100001, 'lambda4': 15200001, 'lambda5': 15300001}
MIX = {regime: {1: .1, 2: .2, 3: .7} for regime in FIRST}


def identities():
    result = []
    for ri, (regime, first) in enumerate(FIRST.items()):
        for case in range(24):
            offset = (ri*24+case) % 7
            for arm in ARMS[offset:]+ARMS[:offset]:
                result.append((regime, first+case, case//3, 1+case%3, arm))
    return result


def rows(candidate=8, control=10, analytic=10):
    result = []
    for regime, seed, block, hit, arm in identities():
        steps = candidate if arm.startswith('gain53') else analytic if arm == 'analytic_inbounds' else control
        result.append({'regime': regime, 'seed': seed, 'block': block, 'initial_hit': hit, 'arm': arm,
                       'steps': steps, 'found': True, 'updates': steps, 'blocked_steps': 0,
                       'final_update_assimilated': True, 'init_seconds': .125, 'choose_seconds': steps/128,
                       'update_seconds': steps/512, 'setup_allocation_seconds': .0625,
                       'controller_seconds': .1875+steps/128+steps/512,
                       'environment_seconds': steps/256, 'state_bytes': 22472})
    return result


def summary(values):
    return S.summarize(values, MIX, FIRST)


def test_independent504_order_rotation_and_each_stratum():
    actual = list(S.evaluation_order(FIRST))
    assert actual == identities() and len(actual) == 504
    for regime in FIRST:
        for arm in ARMS:
            for block in range(8):
                assert sorted(row[3] for row in actual if row[0] == regime and row[2] == block and row[4] == arm) == [1, 2, 3]
    assert [r[-1] for r in actual[168:175]] == list(ARMS[3:]+ARMS[:3])


def test_complete30_gates_cannot_pass_only_relative_improvement():
    good = summary(rows())
    assert good['pilot_continuation'] is True
    assert len(good['competence_checks']) == 18 and len(good['improvement_checks']) == 12
    bad = summary(rows(candidate=100, control=200, analytic=10))
    assert all(c['passes'] for c in bad['improvement_checks'])
    assert sum(c['passes'] for c in bad['competence_checks']) == 9
    assert bad['pilot_continuation'] is False


def test_gain1_competence_is_complete_but_descriptive():
    result = summary(rows(candidate=8, control=20, analytic=10))
    assert result['pilot_continuation'] is True
    controls = result['control_competence_checks']
    assert len(controls) == 18 and sum(c['passes'] for c in controls) == 9
    assert all(not c['passes'] for c in controls if c['name'].endswith('moves'))


def test_relative_move_threshold_inclusive_and_cost_has_no_epsilon():
    result = summary(rows(candidate=95, control=100, analytic=100))
    assert result['pilot_continuation'] is True
    values = rows(candidate=96, control=100, analytic=100)
    assert all(not c['passes'] for c in summary(values)['improvement_checks'] if c['name'].endswith('moves'))
    for row in values:
        row['steps'] = row['updates'] = 95 if row['arm'].startswith('gain53') else 100
        cost = math.nextafter(1., math.inf) if row['arm'].startswith('gain53') else 1.
        row.update(init_seconds=0., update_seconds=0., setup_allocation_seconds=0., choose_seconds=cost, controller_seconds=cost)
    costs = [c for c in summary(values)['improvement_checks'] if c['name'].endswith('.cost')]
    assert len(costs) == 3 and all(not c['passes'] for c in costs)


def test_positive_blocks_are_strict_and_need_six():
    for positive, expected in ((6, True), (5, False)):
        values = rows()
        for row in values:
            if row['arm'].startswith('gain53'):
                row['steps'] = row['updates'] = 8 if row['block'] < positive else 10
        result = summary(values)
        checks = [c for c in result['improvement_checks'] if c['name'].endswith('positive_blocks')]
        assert len(checks) == 3 and all(c['value'] == positive and c['passes'] is expected for c in checks)
        assert result['pilot_continuation'] is expected


def test_hit_mixture_precedes_family_average_and_raw_found_counts():
    values = rows()
    for row in values:
        if row['arm'] == 'gain1@10101':
            row['steps'] = row['updates'] = {1: 10, 2: 20, 3: 30}[row['initial_hit']]
    result = summary(values)
    for regime in FIRST:
        panel = result['regimes'][regime]
        assert panel['means']['gain1@10101']['steps'] == pytest.approx(26.)
        assert panel['family_means']['gain1']['steps'] == pytest.approx(46/3)
        assert panel['raw_counts']['gain1@10101'] == {'found': 24, 'episodes': 24}


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'order', 'nonfinite', 'negative_cost', 'unaccounted_cost',
                                   'long', 'zero', 'bool_steps', 'short_censor', 'missing_update', 'last_update', 'blocked'])
def test_incomplete_or_invalid_episode_is_rejected(defect):
    values = rows()
    if defect == 'missing':
        values.pop()
    elif defect == 'duplicate':
        values[-1] = values[0]
    elif defect == 'order':
        values[0], values[1] = values[1], values[0]
    else:
        key, value = {'nonfinite': ('controller_seconds', math.nan), 'negative_cost': ('environment_seconds', -1.),
                      'unaccounted_cost': ('controller_seconds', 99.), 'long': ('steps', 2189), 'zero': ('steps', 0),
                      'bool_steps': ('steps', True), 'short_censor': ('found', False),
                      'missing_update': ('updates', 0), 'last_update': ('final_update_assimilated', False),
                      'blocked': ('blocked_steps', 1)}[defect]
        values[0][key] = value
    with pytest.raises(ValueError):
        summary(values)


def test_completed_censored_episode_retained_at_full_horizon():
    values = rows()
    values[0].update(found=False, steps=2188, updates=2188)
    result = summary(values)
    assert result['regimes']['lambda3']['raw_counts'][values[0]['arm']]['found'] == 23
    assert result['regimes']['lambda3']['means'][values[0]['arm']]['found'] == pytest.approx(1-.1/8)


def test_wrong_external_plan_pin_stops_before_decoding(tmp_path, monkeypatch):
    path = tmp_path/'plan.json'
    path.write_text('{"untrusted":true}')
    def forbidden(*_):
        raise AssertionError('decoded an unauthenticated plan')
    monkeypatch.setattr(S, 'read', forbidden)
    with pytest.raises(ValueError, match='plan pin'):
        S.authenticate(SimpleNamespace(plan=path, plan_sha256='0'*64))


@pytest.mark.parametrize('channel', ['model_load', 'value_forward', 'parity_restore', 'parity_numpy_forward',
                                    'parity_torch_forward', 'native_step', 'native_reset', 'optimizer_update', 'target_readout'])
def test_all_capped_operations_reject_before_invocation_even_without_resource_check(tmp_path, channel):
    runner = S.Run(SimpleNamespace(output=tmp_path/'run'))
    runner.plan = {'mode': 'qualify', 'limits': S.limits('qualify')}
    mapping = {'native_step': 'native_steps', 'native_reset': 'native_resets',
               'optimizer_update': 'optimizer_updates', 'target_readout': 'target_readouts'}
    cap = runner.plan['limits'][mapping.get(channel, channel)]
    runner.calls[channel] = {'attempted': cap, 'returned': cap, 'seconds': 0.}
    def forbidden():
        raise AssertionError('operation escaped its allocation')
    with pytest.raises(ValueError, match='before invocation'):
        runner.call(channel, forbidden, check=False)
    assert runner.calls[channel]['attempted'] == cap and runner.sequence == 0 and runner.pending == []


def test_amortization_pays_each_fit_prep_sixth_and_load_once(tmp_path, monkeypatch):
    result = summary(rows())
    runner = S.Run(SimpleNamespace(output=tmp_path/'run'))
    receipt = tmp_path/'prior'/'receipt.json'
    runner.plan = {'inputs': {'conditioning_receipt': {'path': str(receipt)}}}
    runner.model_module_setup_seconds = 6.
    fit_costs = {arm: 12.+index for index, arm in enumerate(ARMS[:-1])}
    prior = {'training_costs': fit_costs, 'preparation_seconds': 6., 'initial_pairing_seconds': 99.}
    monkeypatch.setattr(S, 'regular', lambda path: Path(path))
    monkeypatch.setattr(S, 'read', lambda path: {'wall_seconds': 200.} if Path(path).name == 'receipt.json' else prior)
    setup = {arm: {'seconds': 2.} for arm in ARMS[:-1]}
    answer = runner.amortization(result, setup)
    for regime, panel in result['regimes'].items():
        for arm in ARMS:
            row = answer['scenarios'][regime][arm]
            metric = panel['means'][arm]
            if arm == 'analytic_inbounds':
                assert set(row['seconds_per_search'].values()) == {metric['controller_seconds']}
                assert row['fit_seconds'] == row['preparation_share_seconds'] == row['deployment_setup_seconds'] == 0.
            else:
                utility = metric['controller_seconds']-metric['setup_allocation_seconds']
                assert row['deployment_setup_seconds'] == 3. and row['preparation_share_seconds'] == 1.
                assert row['seconds_per_search'] == {str(h): utility+(fit_costs[arm]+4.)/h for h in (1, 100, 10000)}
    assert answer['training_reference']['initial_pairing_seconds'] == 99.  # Disclosed, not added to the formula.


@pytest.mark.parametrize('kind', ['gain1', 'gain53'])
def test_restored_saved_tensors_and_physical_adapter_keep_gain_mass_and_units(tmp_path, monkeypatch, kind):
    import numpy as np
    import torch

    from openjev.research import otto_conditioned_value as model

    torch.set_num_threads(1)
    arrays = {'version': model.VERSION, 'kind': kind, 'input_dim': 11028, 'spatial_gain': model.GAINS[kind],
              'c0': np.asarray(.25, np.float32), **{k: np.zeros(v, np.float32) for k, v in model.SHAPES.items()}}
    arrays['weight_0'][0, 0] = 2.
    arrays['weight_0'][0, -3:] = [.5, -.25, 1.]
    arrays['bias_0'][0], arrays['weight_1'][0, 0], arrays['bias_1'][0] = .125, -2., -.5
    path = tmp_path/'synthetic.npz'
    np.savez(path, **arrays)
    arm = kind+'@10101'
    runner = S.Run(SimpleNamespace(output=tmp_path/'run'))
    runner.np, runner.torch, runner.model = np, torch, model
    runner.plan = {'inputs': {'head_'+arm.replace('@', '_'): {'path': str(path)}}}
    monkeypatch.setattr(S, 'regular', lambda p: Path(p))
    reference = runner.restore_reference(arm)
    assert reference.training is False and reference.spatial_gain == model.GAINS[kind]
    for name, tensor in reference.state_dict().items():
        assert tensor.dtype == torch.float64
        np.testing.assert_array_equal(tensor.numpy(), arrays[name].astype(np.float64))
    centered = np.zeros((3, 105, 105), np.float64)
    centered[0, 0, 0], centered[1, 0, 0] = .5, 1e-12
    positions = np.asarray([[0, 52], [26, 26], [0, 0]], np.int64)
    head = model.FrozenValue(arrays)  # Deliberately no constructor lambda: adapter supplies it per setting.
    for lam in (3., 4., 5.):
        actual = runner.physical(head, centered, positions, lam)
        expected = []
        for field, (x, y) in zip(centered, positions, strict=True):
            mass = float(field.sum())
            hidden = max(2*model.GAINS[kind]*field[0, 0]+.5*mass*x/52-.25*mass*y/52+mass*lam/5+.125, 0)
            expected.append(64*(.25*mass-2*hidden-.5))
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
        with torch.no_grad():
            restored = 64*reference(torch.from_numpy(model.value_features(centered, positions, lam))).numpy()
        np.testing.assert_allclose(actual, restored, atol=1e-10, rtol=1e-10)
    assert actual[-1] == -48.  # No terminal-value override or clipping of the biased zero input.


def test_cleanup_error_preserves_original_failure_and_still_publishes_receipt(tmp_path):
    runner = S.Run(SimpleNamespace(output=tmp_path/'run'))
    runner.bind = lambda: None
    class BadFlush:
        closed = False
        def flush(self):
            raise OSError('flush failed')
        def close(self):
            self.closed = True
    bad = BadFlush()
    runner.handles['fake'] = bad
    runner.pending = [{'id': 1, 'channel': 'parity_numpy_forward'}]
    def fail():
        raise ValueError('primary failure')
    runner.setup = fail
    with pytest.raises(ValueError, match='primary failure') as caught:
        runner.execute()
    assert any('flush failed' in note for note in caught.value.__notes__)
    failure = json.loads((runner.out/'failed.json').read_text())
    assert failure['status'] == 'failed' and failure['pending'] == runner.pending
    assert 'primary failure' in failure['error'] and bad.closed


def test_failed_readout_remains_attempted_pending_and_unreturned(tmp_path):
    runner = S.Run(SimpleNamespace(output=tmp_path/'run'))
    runner.plan = {'mode': 'qualify', 'limits': S.limits('qualify')}
    journal = []
    runner.emit = lambda name, value: journal.append((name, value))
    runner.check = lambda: None
    def fail():
        raise ValueError('synthetic readout failed')
    with pytest.raises(ValueError, match='synthetic readout failed'):
        runner.call('parity_numpy_forward', fail)
    assert runner.calls['parity_numpy_forward'] == {'attempted': 1, 'returned': 0, 'seconds': 0.}
    assert len(runner.pending) == 1 and runner.pending[0]['channel'] == 'parity_numpy_forward'
    assert [value[1] for name, value in journal if name == 'work.jsonl'] == [0]


@pytest.mark.parametrize('found', [True, False])
def test_fake_two_step_episode_pays_one_readout_per_step_and_final_update(tmp_path, monkeypatch, found):
    import numpy as np

    monkeypatch.setattr(S.P, 'HORIZON', 2)
    events, records = [], []
    class Environment:
        position = (26, 26)
        source = np.asarray([40, 40])
        draw_log = ()
        def step(self, action, quiet):
            assert action == 1 and quiet is True
            self.position = (self.position[0]+1, self.position[1])
            events.append(('step', action))
            return 0, 0., found and self.position[0] == 28
    env = Environment()
    class Actor:
        def __init__(self, packet, kernel, allow_stay):
            assert allow_stay is False
            self.public = packet
        def update(self, action, packet):
            events.append(('update', packet['step'], packet['done']))
            self.public = packet
        def storage_bytes(self):
            return {'mutable_array_bytes': 22472}
    branch = SimpleNamespace(raw_masses=np.zeros((4, 4)), weights=np.full((4, 4), 1e-10), eligible_actions=(0, 1, 2, 3))
    def scores(branch, callback, arithmetic):
        assert arithmetic == 'float64'
        callback(np.zeros((16, 1, 1)), np.zeros((16, 2), np.int64), object())
        return np.asarray([4., 1., 3., 2.])
    runner = S.Run(SimpleNamespace(output=tmp_path/'run'))
    runner.np, runner.actor_class, runner.kernels = np, Actor, {'lambda3': object()}
    runner.environment = lambda *_: env
    runner.public = lambda env, step: {'position': env.position, 'hit': 0, 'done': found and step == 2,
                                      'step': step, 'valid_actions': [] if found and step == 2 else [0, 1, 2, 3]}
    def witness(actor, env, packet):
        assert actor.public == packet
        return {'sha256': 'a'*64, 'mass': 1.}
    runner.witness = witness
    runner.branches = SimpleNamespace(rl_branches=lambda *_: branch, explicit_scores=scores,
                                      select_action=lambda costs, allowed: 1)
    # Only the evaluator-side fake actor needs a belief-shaped property here.
    Actor.belief = np.ones((1, 1), np.float64)
    runner.physical = lambda *_: np.ones(16, np.float64)
    def call(channel, operation, **_):
        events.append(('call', channel))
        result = operation()
        runner.last[channel] = {'seconds': 0.}
        return result
    runner.call, runner.check, runner.sync = call, lambda: None, lambda: None
    runner.emit = lambda name, row: records.append((name, row))
    head = SimpleNamespace(storage_bytes=lambda: {'parameter_array_bytes': 8})
    result = runner.episode('lambda3', 15100001, 1, 'gain53@10101', 0, head)
    assert result['steps'] == result['updates'] == 2 and result['found'] is found
    assert result['final_update_assimilated'] is True and result['final_public']['step'] == 2
    assert [event for event in events if event[0] == 'call'] == [('call', 'value_forward'), ('call', 'native_step')]*2
    assert [event for event in events if event[0] == 'update'] == [('update', 1, False), ('update', 2, found)]
    assert [row['kind'] for name, row in records if name == 'eval-transitions.jsonl'] == ['reset', 'step', 'step']
