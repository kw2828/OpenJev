"""Fabricated joint-training roster, rules, admission and lifecycle checks only."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import robot_joint_observer_study as study

ARMS = ('joint_observer_long', 'joint_observer_short', 'continued_last_two', 'continued_temporal')
SEEDS = (8101, 8102, 8103)
ALIAS = 'frozen_position_lr001'
REFERENCES = ('causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence')
OLD_LEARNED = ('local_affine', 'temporal_affine', 'observer_learned')
OLD_FIXED = ('last_two', 'observer_fixed', 'observer_zero')
CACHED = ('joint_local_affine', 'joint_temporal_affine', 'dense_bounded',
          'dense_unbounded', 'gru32', 'legacy_instant', 'gru10')
CONDITIONS = ('complete_forecasts_and_costs', 'equal_four_file_mean_5pct_vs_best_control',
              'each_file_within_2pct_best_simple', 'latency_within_150pct_last_two', 'complete_frontier_not_dominated')


def recipes():
    cfg = study.config(); result = []
    for arm in OLD_LEARNED:
        result += [{'key': f'{arm}-{s}-lr{i}', 'arm': arm, 'seed': s, 'learning_rate': r, 'origin': 'fresh'}
                   for s in SEEDS for i,r in enumerate((.001, .003))]
    for arm in OLD_FIXED:
        result += [{'key': f'{arm}-{s}-fixed', 'arm': arm, 'seed': s, 'learning_rate': None, 'origin': 'fixed'} for s in SEEDS]
    for arm in CACHED:
        result += [{'key': f'{arm}-{s}-cached', 'arm': arm, 'seed': s,
                    'learning_rate': cfg['inherited_rates'][arm], 'origin': 'cached'} for s in SEEDS]
    result += [{'key': a, 'arm': a, 'seed': None, 'learning_rate': None, 'origin': 'reference'} for a in REFERENCES]
    result += [{'key': f'observer_position-{s}-lr{i}', 'arm': 'observer_position', 'seed': s,
                'learning_rate': r, 'origin': 'fresh'} for s in SEEDS for i,r in enumerate((.001, .003))]
    result += [{'key': f'observer_position_fixed-{s}-fixed', 'arm': 'observer_position_fixed', 'seed': s,
                'learning_rate': None, 'origin': 'fixed'} for s in SEEDS]
    result += [{'key': f'{a}-{s}-lr0', 'arm': a, 'seed': s, 'learning_rate': .001, 'origin': 'fresh'} for a in ARMS for s in SEEDS]
    return result


def metrics(value, horizon):
    return {'standardized_rmse': value, 'standardized_sse': value*value*22*horizon*6, 'scalars': 22*horizon*6,
            'physical_rmse_deg': value, 'per_joint_rmse_deg': [value]*6, 'windows': 22, 'horizon': horizon}


def fixture_rows():
    rows = []
    for name in study.EXPOSED:
        for r in recipes():
            for h in (64, 128):
                failed = r['arm'] == 'observer_learned'
                rows.append({'recording': name, 'fit_key': r['key'], **{k: r[k] for k in ('arm', 'seed', 'learning_rate')},
                             'horizon': h, 'status': 'FAILED' if failed else 'PASS',
                             'error': {'type': 'FailedTrainingAttempt'} if failed else None,
                             'metrics': None if failed else metrics(.75 if r['arm'] == ARMS[0] else 1., h)})
    return rows


def cost_roster():
    cfg = study.config(); result = []
    for r in recipes():
        a = r['arm']
        if a in ARMS or a in REFERENCES or (a in cfg['inherited_rates'] and r['learning_rate'] == cfg['inherited_rates'][a]):
            result.append(r.copy())
    result += [{'key': f'{ALIAS}-{s}-lr0', 'arm': ALIAS, 'seed': s, 'learning_rate': .001,
                'origin': 'inherited_alias', 'model_key': f'observer_position-{s}-lr0'} for s in SEEDS]
    return result


def fixture_resources():
    return [{**r, 'status': 'PASS', 'error': None, 'parameter_bytes': 2648, 'buffer_bytes': 0,
             'state_bytes': 48, 'normalizer_bytes': 192, 'timing': {'seconds': [1.]*20, 'median_seconds': 1., 'p95_seconds': 1.}}
            for r in cost_roster()]


def rule(rows=None, resources=None):
    rows = fixture_rows() if rows is None else rows; cfg = study.config()
    return study.evaluate_rule(rows, study.select(rows, cfg), fixture_resources() if resources is None else resources, cfg)


def change(rows, arm, value, *, rate=None, file=None, horizon=None):
    for r in rows:
        if r['arm'] == arm and (rate is None or r['learning_rate'] == rate) and (file is None or r['recording'] == file) and (horizon is None or r['horizon'] == horizon):
            r.update(status='PASS', error=None, metrics=metrics(value, r['horizon']))


def test_exact_rosters_and_passing_witness():
    cfg = study.config(); rows = fixture_rows(); result = rule(rows)
    assert len(recipes()) == 73 and len(rows) == 584
    assert study.identities() == [r for r in recipes() if r['arm'] in ARMS]
    assert [r['arm'] for r in study.identities()] == [a for a in ARMS for _ in SEEDS]
    assert len(study.inherited_identities()) == 45
    assert len(study.resource_identities()) == len(cost_roster()) == 61
    assert {tuple(sorted(r.items())) for r in study.resource_identities()} == {tuple(sorted(r.items())) for r in cost_roster()}
    assert len(study.FAMILIES) == 24 and len(study.ELIGIBLE) == 23
    assert len(cfg['fixed_rates']) == 4 and set(cfg['fixed_rates'].values()) == {.001}
    assert [r['name'] for r in result['conditions']] == list(CONDITIONS)
    assert result['passed'] == result['total'] == 5
    assert result['status'] == 'JOINT_OBSERVER_DEVELOPMENT_PASS'
    assert len(result['equal_file_means']) == 23 and result['best_control_mean'] == 1.


def test_alias_reuses_exact_original_rows_without_duplicate_or_selection():
    rows = fixture_rows(); change(rows, 'observer_position', .5, rate=.001)
    result = rule(rows)
    assert result['equal_file_means'][ALIAS] == .5
    assert result['equal_file_means']['observer_position'] == 1.
    assert result['best_control_mean'] == .5 and not result['conditions'][1]['passed']
    assert not any(r['arm'] == ALIAS for r in rows)
    fixed = study.select(rows, study.config())
    for r in rows:
        if r['recording'] not in study.DEV and r['status'] == 'PASS': r['metrics'] = metrics(100., r['horizon'])
    assert study.select(rows, study.config()) == fixed
    assert fixed['fixed_rates'] == dict.fromkeys(ARMS, .001) and 'options' not in fixed


@pytest.mark.parametrize('candidate,passed', [(.95, True), (.95000001, False)])
def test_mean_relative_boundary(candidate, passed):
    rows = fixture_rows(); change(rows, ARMS[0], candidate)
    assert rule(rows)['conditions'][1]['passed'] is passed


def test_zero_error_control_is_not_relative_gain():
    rows = fixture_rows(); change(rows, ARMS[0], 0.); change(rows, ARMS[1], 0.)
    assert not rule(rows)['conditions'][1]['passed']


@pytest.mark.parametrize('candidate,passed', [(1.02, True), (1.02000001, False)])
def test_file_guard_uses_best_fresh_only(candidate, passed):
    rows = fixture_rows(); change(rows, ARMS[0], candidate, file=study.EXPOSED[0])
    change(rows, 'last_two', .01, file=study.EXPOSED[0])
    assert rule(rows)['conditions'][2]['passed'] is passed


@pytest.mark.parametrize('duration,passed', [(1.5, True), (1.50000001, False)])
def test_latency_uses_concurrent_last_two(duration, passed):
    resources = fixture_resources()
    for r in resources:
        if r['arm'] == ARMS[0]: r['timing']['median_seconds'] = duration
        elif r['arm'] == 'last_two': r['timing']['median_seconds'] = .01
    assert rule(resources=resources)['conditions'][3]['passed'] is passed


def test_frontier_exact_tie_not_dominance_but_one_strict_axis_is():
    rows = fixture_rows(); change(rows, ARMS[1], .75)
    resources = fixture_resources()
    assert rule(rows, resources)['dominators'] == []
    for r in resources:
        if r['arm'] == ARMS[1]: r['parameter_bytes'] -= 4
    assert rule(rows, resources)['dominators'] == [ARMS[1]]


@pytest.mark.parametrize('arm', [*ARMS, 'last_two', 'observer_position', 'causal_ridge_100'])
def test_missing_eligible_h64_cannot_pass_completeness(arm):
    rows = fixture_rows()
    r = next(r for r in rows if r['arm'] == arm and r['horizon'] == 64)
    r.update(status='FAILED', error={'type': 'NonfiniteEvaluation'}, metrics=None)
    assert not rule(rows)['conditions'][0]['passed']


@pytest.mark.parametrize('damage', ['missing_row', 'duplicate_row', 'fake_alias', 'bad_identity', 'bad_geometry', 'nan', 'failure_metrics', 'config'])
def test_score_schema_corruption_rejected(damage):
    rows = fixture_rows(); cfg = study.config()
    if damage == 'missing_row': rows.pop()
    elif damage == 'duplicate_row': rows[-1] = copy.deepcopy(rows[0])
    elif damage == 'fake_alias': rows[-1]['arm'] = ALIAS
    elif damage == 'bad_identity': rows[-1]['learning_rate'] = .003
    elif damage == 'bad_geometry': rows[-1]['metrics']['windows'] = 21
    elif damage == 'nan': rows[-1]['metrics']['standardized_rmse'] = float('nan')
    elif damage == 'failure_metrics': rows[-1]['status'] = 'FAILED'; rows[-1]['error'] = {'type': 'bad'}
    else: cfg['learning_rates'] = [.003]
    with pytest.raises(ValueError): study.select(rows, cfg)


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'rate', 'failed', 'unavailable', 'nan', 'zero', 'negative_bytes'])
def test_cost_omission_or_invalidity_cannot_pass(damage):
    resources = fixture_resources()
    if damage == 'missing': resources.pop()
    elif damage == 'duplicate': resources[-1] = copy.deepcopy(resources[0])
    elif damage == 'rate': next(r for r in resources if r['arm'] == ARMS[0])['learning_rate'] = .003
    elif damage == 'failed': resources[0]['status'] = 'FAILED'; resources[0]['error'] = {'type': 'bad'}
    elif damage == 'unavailable': resources[0].update(status='UNAVAILABLE', error={'type': 'FailedTrainingAttempt'}, timing=None)
    elif damage == 'nan': resources[0]['timing']['median_seconds'] = float('nan')
    elif damage == 'zero': resources[0]['timing']['median_seconds'] = 0.
    else: resources[0]['parameter_bytes'] = -1
    if damage in ('missing', 'duplicate', 'rate'):
        with pytest.raises(ValueError): rule(resources=resources)
    else:
        result = rule(resources=resources)
        assert not result['conditions'][0]['passed'] and not result['frontier_complete']


def descriptor(path):
    data = Path(path).read_bytes(); return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def pin(path):
    return {'path': str(Path(path).resolve()), **descriptor(path)}


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else (json.dumps(value, allow_nan=False)+'\n').encode())
    return pin(path)


def admission_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    for key in study.old.THREADS: monkeypatch.setenv(key, '1')
    sources = {}
    for name in study.SOURCES:
        write(tmp_path/name, ('fake source '+name).encode()); sources[name] = descriptor(tmp_path/name)
    write(tmp_path/study.LAUNCHER, b'fake launcher')
    inputs, inventory = {}, {}
    for i,name in enumerate(sorted(study.expected_input_names())):
        if name.startswith('data/') or name in {'parent/'+n for n in study.COMMON}:
            inputs[name] = write(tmp_path/'opaque'/str(i), ('not an NPZ '+name).encode())
        else:
            relative = name.removeprefix('parent/')
            write(tmp_path/study.publication.STUDY/relative, ('immediate parent '+relative).encode())
            inventory[relative] = descriptor(tmp_path/study.publication.STUDY/relative)
    # Explicit collision: current metadata must override the older same key.
    inputs['parent/fits.json'] = write(tmp_path/'wrong-grandparent-ledger', b'wrong older ledger')
    prior = {'plan': {'inputs': inputs}, 'inventory': inventory, 'audit': {'results': {
        'result': {'status': 'POSITION_OBSERVER_DEVELOPMENT_FAIL'},
        'selection': {'selected_rates': {'observer_position': .003}}}}}
    visits = []
    def closed_parent(*args): visits.append('closed-parent'); return prior
    monkeypatch.setattr(study.publication, 'authenticate', closed_parent)
    def forbidden(*args, **kwargs): raise AssertionError('decoder before admission')
    monkeypatch.setattr(study.np, 'load', forbidden)
    q = {'status': 'PASS', 'sources': sources, 'launcher': descriptor(tmp_path/study.LAUNCHER),
         'sources_unchanged': True, 'thread_env': dict.fromkeys(study.old.THREADS, '1'), 'commands': []}
    commands = [['.venv/bin/ruff', 'check', *[p for p in study.QUALIFICATION_SOURCES if p.endswith('.py')]],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', 'tests/test_robot_joint_observer.py',
                 'tests/test_robot_joint_observer_study.py', 'tests/test_audit_robot_joint_observer.py']]
    for i,command in enumerate(commands):
        log = tmp_path/f'qualification-{i}.log'; write(log, b'fabricated PASS')
        q['commands'].append({'command': command, 'returncode': 0, 'log': str(log), 'sha256': descriptor(log)['sha256'], 'seconds': 1.})
    qpath = tmp_path/'qualification.json'; write(qpath, q)
    plan = {'version': study.VERSION, 'config': study.config(), 'sources': sources,
            'launcher': descriptor(tmp_path/study.LAUNCHER), 'qualification': pin(qpath),
            'inputs': study.expected_inputs(prior),
            'parent_audit': write(tmp_path/study.publication.AUDIT, b'opaque parent audit'),
            'parent_publication_manifest': write(tmp_path/study.publication.OUTPUT/'manifest.json', b'opaque public manifest')}
    path = tmp_path/'registration.json'; write(path, plan)
    return path, plan, qpath, q, prior, visits


def test_actual_admission_of_opaque_exact63_source69_input_fixture(tmp_path, monkeypatch):
    path, plan, _, _, prior, visits = admission_fixture(tmp_path, monkeypatch)
    actual, sha = study.authenticate(path)
    assert actual == plan and sha == descriptor(path)['sha256'] and visits == ['closed-parent']
    assert len(plan['sources']) == 63 and len(plan['inputs']) == 69
    assert sum(k.startswith('data/') for k in plan['inputs']) == 11
    assert sum(k.endswith('/final.npz') for k in plan['inputs']) == 45
    assert plan['inputs']['parent/fits.json'] != prior['plan']['inputs']['parent/fits.json']
    assert not any('22H_58M' in k for k in plan['inputs'])


@pytest.mark.parametrize('damage', ['source', 'source_roster', 'launcher', 'config', 'qualification_status', 'qualification_sources',
    'qualification_unchanged', 'qualification_command', 'qualification_code', 'qualification_log', 'threads', 'input_pin',
    'input_roster', 'parent_audit', 'parent_manifest', 'parent_closure', 'parent_selection', 'parent_inventory'])
def test_admission_damage_blocks_before_decoder(tmp_path, monkeypatch, damage):
    path, plan, qpath, q, prior, _ = admission_fixture(tmp_path, monkeypatch)
    if damage == 'source': write(tmp_path/study.SOURCES[0], b'changed')
    elif damage == 'source_roster': plan['sources'].pop(study.SOURCES[0])
    elif damage == 'launcher': write(tmp_path/study.LAUNCHER, b'changed')
    elif damage == 'config': plan['config']['fixed_rates'][ARMS[0]] = .003
    elif damage == 'qualification_status': q['status'] = 'FAILED'
    elif damage == 'qualification_sources': q['sources'] = {}
    elif damage == 'qualification_unchanged': q['sources_unchanged'] = False
    elif damage == 'qualification_command': q['commands'][1]['command'][-1] = 'tests/other.py'
    elif damage == 'qualification_code': q['commands'][0]['returncode'] = 1
    elif damage == 'qualification_log': write(q['commands'][0]['log'], b'changed')
    elif damage == 'threads': monkeypatch.setenv(study.old.THREADS[0], '2')
    elif damage == 'input_pin': write(next(iter(plan['inputs'].values()))['path'], b'changed')
    elif damage == 'input_roster': plan['inputs'].pop(next(iter(plan['inputs'])))
    elif damage == 'parent_audit': write(plan['parent_audit']['path'], b'changed')
    elif damage == 'parent_manifest': write(plan['parent_publication_manifest']['path'], b'changed')
    elif damage == 'parent_closure':
        def reject(*args): raise ValueError('parent still live')
        monkeypatch.setattr(study.publication, 'authenticate', reject)
    elif damage == 'parent_selection': prior['audit']['results']['selection']['selected_rates']['observer_position'] = .001
    else: prior['inventory']['fits.json'] = {'bytes': 0, 'sha256': '0'*64}
    if damage.startswith('qualification_') and damage != 'qualification_log': write(qpath, q); plan['qualification'] = pin(qpath)
    write(path, plan)
    with pytest.raises(ValueError): study.authenticate(path)


def test_loader_pin_before_decode_and_k_order(tmp_path, monkeypatch):
    path = tmp_path/'toy.npz'; values = np.asfortranarray(np.arange(30, dtype=np.float32).reshape(6, 5))
    np.savez_compressed(path, values=values)
    plan = {'inputs': {'parent/normalizers.npz': pin(path)}}
    result = study.load(plan, 'parent/normalizers.npz')['values']
    assert result.flags.f_contiguous and not result.flags.c_contiguous
    np.testing.assert_array_equal(result, values)
    write(path, b'not NPZ')
    def forbidden(*a, **k): raise AssertionError('NPZ decoder before pin')
    monkeypatch.setattr(study.np, 'load', forbidden)
    with pytest.raises(ValueError): study.load(plan, 'parent/normalizers.npz')
    with pytest.raises(ValueError): study.load(plan, 'data/official_TEST.mat')


def test_inherited_window_alignment():
    t = np.arange(300, dtype=np.float64)[:, None]
    record = {'q': np.repeat(t, 6, axis=1), 'u': np.repeat(t+10000., 6, axis=1)}
    b = study.old.window_batch([record], {'record': np.array([0]), 'start': np.array([64])}, 32, 128)
    np.testing.assert_array_equal(b['q_context'][0, :, 0], np.arange(64, 96))
    np.testing.assert_array_equal(b['future_u'][0, :, 0], np.arange(10095, 10223))
    np.testing.assert_array_equal(b['target'][0, :, 0], np.arange(96, 224))
    assert list(study.old.dev_windows(3636, study.config())) == list(64+160*np.arange(22))


def toy_model():
    """Only algebraic parameters, no robot model or saved checkpoint."""
    import torch
    model = torch.nn.Module(); model.cell = torch.nn.Module()
    model.cell.coefficient = torch.nn.Parameter(torch.zeros(590))
    model.gain = torch.nn.Parameter(torch.cat((torch.eye(6), torch.zeros(6, 6))))
    return model


def fake_infer(model, batch, prefix):
    prefix.update(prefix_steps=30, max_state_abs=2., max_state_norm64=3., max_innovation_abs=1., max_innovation_norm64=2.)
    return sum(p.sum() for p in model.parameters()).expand(1, 128, 6)


def test_generic_gradient_snapshot_includes_cell_and_gain_before_clip():
    import torch
    model = toy_model(); sum(p.sum()*2 for p in model.parameters()).backward()
    before = study.gradient_snapshot(model)
    assert set(before) == {'gain', 'cell.coefficient'}
    assert sum(a.size for a in before.values()) == 662
    for a in before.values(): np.testing.assert_array_equal(a, np.full_like(a, 2.))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert all(np.max(a) == 2. for a in before.values())
    assert all(float(p.grad.abs().max()) < .04 for p in model.parameters())
    model.cell.coefficient.grad = None
    with pytest.raises(ValueError, match='present gradient'): study.gradient_snapshot(model)


def test_initialize_copies_only_common_cell_preserves_head_and_trainability():
    import torch
    model = toy_model(); original = {'cell.coefficient': np.arange(590, dtype=np.float32)}
    initial = study.initialize(model, original)
    np.testing.assert_array_equal(model.cell.coefficient.detach().numpy(), original['cell.coefficient'])
    assert torch.equal(model.gain, torch.cat((torch.eye(6), torch.zeros(6, 6))))
    assert initial['optimizer_state_entries'] == 0 and all(p.requires_grad for p in model.parameters())
    assert initial['cell_sha256'] == study.parent.cell_hash(model)
    with pytest.raises(ValueError): study.initialize(model, {'gain': np.zeros((12, 6), np.float32), **original})


@pytest.mark.parametrize('failure', [None, 'numerical', 'programming'])
def test_tiny_training_has_fresh_adam_all_parameters_and_preserved_attempts(tmp_path, monkeypatch, failure):
    model = toy_model(); cfg = {**study.config(), 'updates': 2, 'batch_size': 1}
    batch = {'target': np.full((1, 128, 6), -1., dtype=np.float64)}
    monkeypatch.setattr(study.old, 'window_batch', lambda *a, **k: batch)
    visits = []; original_adam = study.torch.optim.Adam
    def fresh(parameters, **kwargs):
        parameters = tuple(parameters); optimizer = original_adam(parameters, **kwargs)
        assert len(optimizer.state) == 0 and sum(p.numel() for p in parameters) == 662
        visits.append('empty-Adam'); return optimizer
    monkeypatch.setattr(study.torch.optim, 'Adam', fresh)
    calls = []
    error = ValueError('programming problem')
    def infer(m, b, prefix):
        calls.append(1)
        if len(calls) == 2 and failure == 'numerical': raise study.old.FitFailure('nonfinite autoregressive training loss')
        if len(calls) == 2 and failure == 'programming': raise error
        return fake_infer(m, b, prefix)
    monkeypatch.setattr(study, 'infer_diagnostic', infer)
    folder = tmp_path/'fit'
    if failure == 'programming':
        with pytest.raises(ValueError) as caught:
            study.train_one(model, [], {'record': np.zeros((2, 1), np.int64)}, cfg=cfg, lr=.001, folder=folder)
        assert caught.value is error
        receipt = study.read(folder/'fit-receipt.json')
    else:
        receipt = study.train_one(model, [], {'record': np.zeros((2, 1), np.int64)}, cfg=cfg, lr=.001, folder=folder)
    assert visits == ['empty-Adam'] and len(calls) == 2
    steps = 2 if failure is None else 1
    assert receipt['completed_updates'] == steps
    assert receipt['status'] == ({None: 'PASS', 'numerical': 'FAILED', 'programming': 'FATAL'}[failure])
    assert set(receipt['files']) == {'initial.npz', 'final.npz', 'optimizer.npz', 'trace.json',
                                    'diagnostics.json', 'diagnostic-summary.json', 'last-gradient.npz'}
    with np.load(folder/'optimizer.npz') as bank:
        arrays = {k: bank[k] for k in bank.files}
    study.check_optimizer(model, arrays, steps)
    assert set(arrays) == {n+'/'+key for n in ('gain', 'cell.coefficient') for key in ('step', 'exp_avg', 'exp_avg_sq')}
    summaries = study.read(folder/'diagnostic-summary.json')
    assert summaries['attempts'] == 2 and summaries['backward_calls'] == summaries['clip_calls'] == steps
    with np.load(folder/'last-gradient.npz') as bank:
        assert set(bank.files) == ({'gain', 'cell.coefficient', 'native_norm'} if failure is None else set())


@pytest.mark.parametrize('whole', [False, True])
def test_native_fit_cap_retained_but_whole_cap_fatal(tmp_path, monkeypatch, whole):
    class Clock:
        def now_ns(self): return 2_000_000_000
    def check(): study.history.check_deadline(Clock(), 0, 1., whole=whole)
    model = toy_model(); folder = tmp_path/'fit'
    if whole:
        with pytest.raises(TimeoutError):
            study.train_one(model, [], {}, cfg=study.config(), lr=.001, folder=folder, check=check)
        assert study.read(folder/'fit-receipt.json')['status'] == 'FATAL'
    else:
        receipt = study.train_one(model, [], {}, cfg=study.config(), lr=.001, folder=folder, check=check)
        assert receipt['status'] == 'FAILED' and receipt['completed_updates'] == 0


def test_failed_training_cost_never_calls_timer_and_alias_uses_underlying_model(monkeypatch):
    identity = study.identities()[0]; spec = {'parameter_bytes': 2648, 'buffer_bytes': 0, 'state_bytes': 48, 'normalizer_bytes': 192}
    lookup = {identity['key']: {'effective_status': 'FAILED', 'resources': spec}}
    def forbidden(*a, **k): raise AssertionError('failed fit was timed')
    result = study.timing_attempt(identity, lookup, forbidden, None, None, study.config())
    assert result['status'] == 'UNAVAILABLE' and result['timing'] is None and result['parameter_bytes'] == 2648
    alias = study.alias_identities()[0]; seen = []
    lookup[alias['model_key']] = {'effective_status': 'PASS', 'resources': spec}
    def predictor(r): seen.append(r['model_key']); return object()
    monkeypatch.setattr(study.history, 'timed_request', lambda *a: {'median_seconds': .5})
    result = study.timing_attempt(alias, lookup, predictor, None, None, study.config())
    assert seen == ['observer_position-8101-lr0'] and result['arm'] == ALIAS and result['status'] == 'PASS'


@pytest.mark.parametrize('known', [True, False])
def test_timing_preserves_known_guard_and_raises_programming_errors(monkeypatch, known):
    identity = study.identities()[0]; lookup = {identity['key']: {'effective_status': 'PASS', 'resources': {}}}
    error = ValueError('nonfinite ridge prediction' if known else 'unexpected schema error')
    def timer(*a): raise error
    monkeypatch.setattr(study.history, 'timed_request', timer)
    if known:
        result = study.timing_attempt(identity, lookup, lambda r: object(), None, None, study.config())
        assert result['status'] == 'FAILED' and result['timing'] is None
    else:
        with pytest.raises(ValueError) as caught: study.timing_attempt(identity, lookup, lambda r: object(), None, None, study.config())
        assert caught.value is error


def test_all_twelve_failed_attempts_and57_checkpoints_before_exposed_decode(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    cfg, input_paths, events = study.config(), {}, []
    original_rows = [r for r in fixture_rows() if r['arm'] not in ARMS]
    original_attempts = [{'opaque_parent_attempt': i} for i in range(244)]
    inherited = [{**r, 'effective_status': 'PASS', 'resources': {}} for r in study.inherited_identities()]
    payloads = {'parent/results.json': {'rows': original_rows}, 'parent/prediction-attempts.json': original_attempts,
                'parent/fits.json': inherited, 'parent/causal_ridge_1.json': {'fit_rows': 1},
                'parent/causal_ridge_100.json': {'fit_rows': 1}}
    for name, value in payloads.items(): input_paths[name] = write(tmp_path/'inputs'/name, value)
    for r in study.inherited_identities():
        name = 'parent/'+r['key']+'/final.npz'; input_paths[name] = write(tmp_path/'inputs'/name, b'opaque inherited checkpoint')
    registration = tmp_path/'registration.json'; write(registration, {})
    plan = {'sources': {}, 'inputs': input_paths}
    monkeypatch.setattr(study, 'authenticate', lambda path: (plan, 'fabricated-sha'))
    monkeypatch.setattr(study, 'SOURCES', ())
    bank = {'record': np.zeros((4096, 16), np.int64), 'start': np.zeros((4096, 16), np.int64)}
    class AtExposedBoundary(RuntimeError): pass
    def load(plan, name):
        if name.startswith('data/'):
            if name.removeprefix('data/') in study.EXPOSED:
                barrier = study.read(tmp_path/'out/checkpoint-barrier.json')
                assert barrier['fresh_fit_attempts'] == 12 and barrier['inherited_models'] == 45
                assert len(barrier['checkpoints']) == len(barrier['state_sha256']) == 57
                assert barrier['exposed_loads_this_run'] == 0
                assert events == [r['key'] for r in study.identities()]
                raise AtExposedBoundary('intentional no measured decode')
            return {'q': np.zeros((300, 6)), 'u': np.zeros((300, 6))}
        if name == 'parent/normalizers.npz': return {'q_mean': np.zeros(6), 'q_std': np.ones(6), 'u_mean': np.zeros(6), 'u_std': np.ones(6)}
        if name == 'parent/linear.npz': return {'coefficient': np.zeros((6, 25))}
        if 'batches-' in name: return bank
        if 'causal_ridge_' in name: return {f'h{h:03d}': np.zeros((1, 6)) for h in range(1, 129)}
        return {'cell.coefficient': np.zeros(590, np.float32)}
    monkeypatch.setattr(study, 'load', load)
    monkeypatch.setattr(study.old, 'normalized_record', lambda record, norm: record)
    monkeypatch.setattr(study.old, 'make_batches', lambda *a: bank)
    monkeypatch.setattr(study, 'CausalRobotRidge', lambda *a: SimpleNamespace())
    def model_for(*a):
        model = toy_model(); model.load_state_dict = lambda *a, **k: None; return model
    monkeypatch.setattr(study, 'model_for', model_for)
    monkeypatch.setattr(study, 'resource_model', lambda *a: {})
    def train(model, records, batches, **kwargs):
        assert batches is bank and kwargs['lr'] == .001 and kwargs['cfg'] == cfg
        folder = kwargs['folder']; folder.mkdir()
        np.savez_compressed(folder/'final.npz', **study.old.weights(model))
        events.append(folder.name)
        return {'status': 'FAILED', 'completed_updates': 0}
    monkeypatch.setattr(study, 'train_one', train)
    with pytest.raises(AtExposedBoundary): study.run(registration, tmp_path/'out')
    fits = study.read(tmp_path/'out/fits.json')
    assert len(fits) == 57 and sum(r['effective_status'] == 'FAILED' for r in fits) == 12
    assert study.read(tmp_path/'out/parent/results.json')['rows'] == original_rows
    assert study.read(tmp_path/'out/parent/prediction-attempts.json') == original_attempts
    assert study.read(tmp_path/'out/failure.json')['type'] == 'AtExposedBoundary'


@pytest.mark.parametrize('damage', ['source', 'qualification', 'uncommitted', 'preexisting'])
def test_launcher_guard_before_job_record_and_child(tmp_path, monkeypatch, damage):
    import launch_robot_joint_observer_study as launch
    registration, plan, qpath, q, _, _ = admission_fixture(tmp_path, monkeypatch)
    copied = tmp_path/study.LAUNCHER; copied.write_bytes(Path(launch.__file__).read_bytes())
    plan['launcher'] = descriptor(copied); q['launcher'] = plan['launcher']
    engineering = tmp_path/'engineering'; engineering.mkdir(); output = tmp_path/'out'
    if damage == 'source': write(tmp_path/study.SOURCES[0], b'changed')
    elif damage == 'qualification': q['status'] = 'FAILED'
    elif damage == 'preexisting': output.mkdir()
    write(qpath, q); plan['qualification'] = pin(qpath); write(registration, plan)
    monkeypatch.setattr(launch, 'ROOT', tmp_path); monkeypatch.setattr(launch, 'REGISTRATION', registration)
    monkeypatch.setattr(launch, 'ENGINEERING', engineering); monkeypatch.setattr(launch, 'OUTPUT', output)
    def git(command, **kwargs):
        if command == ['git', 'rev-parse', 'HEAD']: return 'f'*40
        name = command[2].split(':', 1)[1]
        return b'uncommitted' if damage == 'uncommitted' else (tmp_path/name).read_bytes()
    def forbidden(*a, **k): raise AssertionError('child launched after failed metadata gate')
    monkeypatch.setattr(launch.subprocess, 'check_output', git)
    monkeypatch.setattr(launch.subprocess, 'run', forbidden)
    with pytest.raises(ValueError): launch.main()
    assert not (engineering/'run-launch-01.json').exists() and not (engineering/'run-process-01.json').exists()
