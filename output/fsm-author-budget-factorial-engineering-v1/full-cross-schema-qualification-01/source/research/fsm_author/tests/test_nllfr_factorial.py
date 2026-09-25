# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic contracts only; no measured files, fitted models or real requests."""
import copy
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent/'src'))
sys.path.insert(0, str(HERE.parent/'scripts'))
import evaluate_nllfr_factorial as study  # noqa: E402


def legal(start=0):
    return {'starts': np.array([start], dtype=np.int64), 'y_context': np.arange(300, dtype=np.float64).reshape(1, 100, 3),
        'u_context': np.arange(3, 300, dtype=np.float64).reshape(1, 99, 3),
        'future_u': np.arange(300, 684, dtype=np.float64).reshape(1, 128, 3)}


def result(value=1., status='GRADIENT_TOL'):
    state = np.ones((1, 1))
    return {'prediction': np.full((1, 128, 3), value), 'final_state': state.copy(), 'forecast_state': state.copy(),
        'context': {'requests': [{'status': status, 'trace': [{'iteration': 0, 'trials': []}],
            'final_objective': 1., 'directions_considered': 1, 'accepted_steps': 0,
            'jacobian_evaluations': 2, 'trajectory_evaluations': 2, 'trial_attempts': 0}],
            'linear_seed_states': state.copy(), 'solved_context_start_states': state.copy()},
        'timing': {'request_ms': 4., 'copy_ms': 1., 'initializer_ms': 2., 'rollout_ms': 1.}}


def previous():
    means = {'candidate': .5, 'strong': 1., 'other': 2.}
    records = {k: dict.fromkeys(study.DEV_IDS, v) for k, v in means.items()}
    return {'candidate': 'candidate', 'family_means': means, 'per_record_means': records,
        'per_seed_means': {'candidate': {'9201': .5, '9202': .5, '9203': .5}, 'strong': {'None': 1.}, 'other': {'None': 2.}},
        'per_amplitude_means': {}, 'conditions': dict.fromkeys(('five_percent_below_strongest_control',
            'every_seed_below_strongest_control', 'no_record_over_two_percent_strongest_control',
            'both_amplitudes_below_strongest_control'), False)}


def cells(means=(2., 1.5, 1., .75)):
    return {cell: {'mean_rmse': v, 'median_request_ms': 3.,
        'rows': [{'record_id': rid, 'status': 'complete', 'rmse': v} for rid in study.DEV_IDS]}
        for cell, v in zip(study.CELLS, means, strict=True)}


def test_exact_full_roster_and_balanced_rotated_timing_schedule():
    assert len(study.DEV_IDS) == 12 and study.STARTS == tuple(range(0, 7937, 256))
    assert len(study.DEV_IDS)*len(study.STARTS)*len(study.CELLS) == 1536
    rows = study.timing_schedule()
    assert len(rows) == 96 and Counter(r['cell'] for r in rows) == dict.fromkeys(study.CELLS, 24)
    for cell in study.CELLS:
        assert Counter(r['position'] for r in rows if r['cell'] == cell) == dict.fromkeys(range(4), 6)
    assert [r['cell'] for r in rows[:4]] == list(study.CELLS)
    assert [r['cell'] for r in rows[4:8]] == ['old64', 'new16', 'new64', 'old16']
    assert {(r['record_id'], r['start']) for r in rows} == {(r, s) for r in study.DEV_IDS for s in (0, 7936)}


def test_full_request_timer_starts_before_copies_and_covers_condition_and_forecast(monkeypatch):
    clock, seen = iter([10., 11., 13., 14.]), []
    inputs = legal()
    def tick():
        seen.append('clock')
        return next(clock)
    def condition(_model, y, u, *, iterations):
        assert seen == ['clock', 'clock'] and iterations == 64
        assert not np.shares_memory(y, inputs['y_context']) and not np.shares_memory(u, inputs['u_context'])
        np.testing.assert_array_equal(u[0, 0], [3., 4., 5.])
        y[:] = -5.
        seen.append('condition')
        return np.ones((1, 1)), {'only': 'context'}
    def rollout(_model, future, state):
        assert seen[-1] == 'clock' and not np.shares_memory(future, inputs['future_u'])
        np.testing.assert_array_equal(future[0, 0], [300., 301., 302.])
        seen.append('forecast')
        return np.zeros((1, 128, 3)), state.copy()
    monkeypatch.setattr(study.time, 'perf_counter', tick)
    monkeypatch.setattr(study, 'condition', condition)
    monkeypatch.setattr(study, 'physical_rollout', rollout)
    answer = study.request({}, inputs, 64)
    assert answer['timing'] == {'request_ms': 4000., 'copy_ms': 1000., 'initializer_ms': 2000., 'rollout_ms': 1000.}
    np.testing.assert_array_equal(inputs['y_context'][0, 0], [0., 1., 2.])


@pytest.mark.parametrize('bad', [None, 'start', 'dtype', 'shape', 'nonfinite', 'extra'])
def test_legal_cache_exact_keys_readonly_owned_and_no_targets(tmp_path, bad):
    values = legal(256)
    if bad == 'start':
        values['starts'][0] = 0
    elif bad == 'dtype':
        values['y_context'] = values['y_context'].astype(np.float32)
    elif bad == 'shape':
        values['u_context'] = values['u_context'][:, :-1]
    elif bad == 'nonfinite':
        values['future_u'][0, 0, 0] = np.nan
    elif bad == 'extra':
        values['target'] = np.zeros((1, 128, 3))
    path = tmp_path/'legal.npz'
    np.savez(path, **values)
    if bad is not None:
        with pytest.raises(ValueError):
            study.legal_cache(path, 256)
    else:
        actual = study.legal_cache(path, 256)
        for key in ('y_context', 'u_context', 'future_u'):
            assert not actual[key].flags.writeable and not np.shares_memory(actual[key], values[key])
        assert set(actual) == set(study.LEGAL_KEYS)


def test_standardized_scoring_has_hand_closed_form_and_native_units():
    y = np.array([[[1., 2., 3.], [1., 2., 3.]]])
    row = study.standardized_score(y, np.zeros_like(y), np.array([2., 3., 4.]))
    assert row['mse'] == pytest.approx(14/3) and row['rmse'] == pytest.approx(np.sqrt(14/3))
    assert row['per_channel_rmse'] == [1., 2., 3.] and row['native_output_per_channel_rmse'] == [2., 6., 12.]


@pytest.mark.parametrize('native', [False, True])
def test_known_standardized_and_native_score_overflow_retained_as_numeric(native):
    prediction = np.ones((1, 2, 3))*(2. if native else 1e200)
    with pytest.raises(ValueError, match='score overflow') as got:
        study.standardized_score(prediction, np.zeros_like(prediction), np.full(3, 1e308 if native else 1.))
    assert study.prior.numerical_failure(got.value)


def test_prefix_requires_same_seed_trace_monotonicity_and_early_stop_identity():
    short = result()
    long = copy.deepcopy(short)
    assert study.prefix_check(short, long)['status'] == 'PASS'
    long['prediction'][0, 0, 0] += 1
    assert study.prefix_check(short, long)['checks']['early_stop_identical'] is False
    for mutation in ('seed', 'trace', 'objective'):
        long = copy.deepcopy(short)
        if mutation == 'seed':
            long['context']['linear_seed_states'] += 1
        elif mutation == 'trace':
            long['context']['requests'][0]['trace'][0]['iteration'] = 1
        else:
            long['context']['requests'][0]['final_objective'] = 2.
        assert study.prefix_check(short, long)['status'] == 'FAILED'
    assert study.prefix_check(None, long)['status'] == 'unavailable'


def test_capped_prefix_allows_worse_forecast_but_not_worse_context():
    short = result(status='ITERATION_CAP')
    long = copy.deepcopy(short)
    long['prediction'] += 100.
    long['context']['requests'][0]['trace'].append({'iteration': 1, 'trials': []})
    long['context']['requests'][0]['final_objective'] = .5
    assert study.prefix_check(short, long)['status'] == 'PASS'


def test_old16_exact_replay_not_tolerance_or_score_equivalence():
    value, inputs = result(), legal()
    saved = {'prediction': value['prediction'].copy(), **inputs,
        **{k: (value[k] if k in value else value['context'][k]).copy() for k in study.STATE_KEYS}}
    row = {'context': study.json_tree(value['context'])}
    assert study.old_replay_check(value, saved, row, value['prediction'], inputs)['status'] == 'PASS'
    saved['final_state'][0, 0] = np.nextafter(1., 2.)
    assert study.old_replay_check(value, saved, row, value['prediction'], inputs)['status'] == 'FAILED'


def test_contrast_sign_denominators_and_interaction_not_forecast_promise():
    value = study.contrasts(cells())
    effects = value['effects']
    assert effects['training_at16']['absolute_reduction'] == 1.
    assert effects['training_at16']['relative_reduction_percent'] == 50.
    assert effects['training_at64']['absolute_reduction'] == .75
    assert effects['inference_old']['absolute_reduction'] == .5
    assert effects['inference_new']['absolute_reduction'] == .25
    assert value['interaction_absolute'] == -.25
    altered = cells((0., 0., 0., 0.))
    assert study.contrasts(altered)['effects']['training_at16']['relative_reduction_percent'] is None
    altered['new16']['mean_rmse'] = None
    assert study.contrasts(altered)['interaction_absolute'] is None


def test_quality_controls_survive_missing_cost_but_global_continuation_is_blocked():
    data = cells()
    expected = study.continuation(previous(), data, True, True)
    assert expected['passed'] == 4 and expected['strongest_control'] == 'new64'
    data['new64']['median_request_ms'] = None
    answer = study.continuation(previous(), data, True, True)
    assert answer['strongest_control'] == 'new64' and answer['passed'] == 0
    assert answer['family_means']['new64'] == .75
    for fit_complete, parity in ((False, True), (True, False)):
        answer = study.continuation(previous(), cells(), fit_complete, parity)
        assert answer['passed'] == 0 and not answer['reference_complete']
    data = cells()
    data['old16']['mean_rmse'] = None
    assert study.continuation(previous(), data, True, True)['passed'] == 0


@pytest.mark.parametrize('error', [FloatingPointError('overflow'), ValueError('score overflow')])
def test_timing_known_failure_preserved_and_programming_error_propagates(monkeypatch, error):
    def fail(*args):
        raise error
    monkeypatch.setattr(study, 'request', fail)
    row = study.timed_attempt({'cell': 'new64'}, {}, {})
    assert row['status'] == 'failed' and row['error']['type'] == type(error).__name__
    assert 'request_ms' not in row
    def bug(*args):
        raise ValueError('programming defect')
    monkeypatch.setattr(study, 'request', bug)
    with pytest.raises(ValueError, match='programming defect'):
        study.timed_attempt({'cell': 'new64'}, {}, {})


@pytest.mark.parametrize('failure', [None, 'forecast', 'warmup', 'timing', 'merged'])
def test_fake_full_lifecycle_retains1536_slots_and96_timings_even_after_warmup_failure(tmp_path, monkeypatch, failure):
    cache = {(rid, s): legal(s) for rid in study.DEV_IDS for s in study.STARTS}
    models = {k: {'A': np.ones((1, 1))} for k in ('old', 'new')}
    calls, saves, returned = [], [], []
    def fake_request(_model, inputs, iterations):
        index = len(calls)
        calls.append((int(inputs['starts'][0]), iterations))
        if (failure, index) in {('forecast', 0), ('warmup', 1536), ('timing', 1540)}:
            raise FloatingPointError('synthetic numerical failure')
        returned.append(len(calls))
        return result()
    class Archive:
        def __init__(self, path):
            self.start = int(Path(path).stem)
        def __enter__(self):
            assert returned[-1] == len(calls)  # Target read follows this successful return.
            return self
        def __exit__(self, *args):
            return False
        def __getitem__(self, key):
            if key == 'target':
                return np.zeros((1, 128, 3))
            if key in legal(self.start):
                return legal(self.start)[key]
            a = result()
            return a[key] if key in a else a['context'][key]
    original_read = study.read
    def fake_read(path):
        if Path(path).name.endswith('.json') and 'reference' in Path(path).parts:
            return {'context': study.json_tree(result()['context'])}
        return original_read(path)
    monkeypatch.setattr(study, 'request', fake_request)
    monkeypatch.setattr(study.np, 'load', lambda path, **kwargs: Archive(path))
    monkeypatch.setattr(study.np, 'savez_compressed', lambda path, **values: saves.append((str(path), set(values))))
    monkeypatch.setattr(study, 'read', fake_read)
    monkeypatch.setattr(study, 'event', lambda *a, **k: None)
    monkeypatch.setattr(study, 'rescore_references', lambda *a: {'rescored_continuation': previous()})
    original_score, rejected_merged = study.standardized_score, []
    def fake_score(prediction, target, scale):
        if failure == 'merged' and len(prediction) == 32 and not rejected_merged:
            rejected_merged.append(True)
            raise ValueError('score overflow')
        return original_score(prediction, target, scale)
    monkeypatch.setattr(study, 'standardized_score', fake_score)
    answer = study.evaluate(models, cache, {'y_mean': np.zeros(3), 'y_scale': np.ones(3)},
        tmp_path/'reference', tmp_path, {'prior_continuation': previous(), 'new_fit_status': 'FIT_ONLY_COMPLETE'})
    assert len(calls) == 1536+4+96
    assert sum(len(v['requests']) for v in answer['cells'].values()) == 1536
    assert sum(len(v['rows']) for v in answer['cells'].values()) == 48
    assert len(answer['parity']['old16']) == 384 and len(answer['parity']['within_checkpoint']) == 768
    assert len(answer['timing']['samples']) == 96 and len(answer['timing']['warmups']) == 4
    assert all(sum(r['cell'] == c for r in answer['timing']['samples']) == 24 for c in study.CELLS)
    if failure == 'forecast':
        assert answer['cells']['old16']['rows'][0]['completed_requests'] == 31
        assert answer['cells']['old16']['mean_rmse'] is None and not answer['parity_pass']
    if failure in ('warmup', 'timing'):
        assert answer['cells']['old16']['median_request_ms'] is None
        assert answer['cells']['old16']['mean_rmse'] == 1.
    if failure == 'merged':
        row = answer['cells']['old16']['rows'][0]
        assert row['status'] == 'incomplete' and row['completed_requests'] == 32
        assert row['error']['message'] == 'score overflow'
        assert answer['cells']['old16']['mean_rmse'] is None
        assert any(p.endswith('/'+study.DEV_IDS[0]+'.npz') for p, _ in saves)


def test_reference_rescore_equal_seed_record_means_and_exact_shared_targets(monkeypatch):
    values = {'candidate-1': 1., 'candidate-2': 2., 'candidate-3': 3., 'strong-None': 4.}
    banks = [{'family': family, 'seed': seed, 'record_id': rid, 'file': {'path': f'{family}-{seed}/{rid}.npz'}}
        for family, seeds in (('candidate', [1, 2, 3]), ('strong', [None]))
        for seed in seeds for rid in study.DEV_IDS]
    targets = {rid: np.zeros((32, 128, 3)) for rid in study.DEV_IDS}
    class Archive:
        def __init__(self, path):
            self.value = values[Path(path).parent.name]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def __getitem__(self, key):
            return {'prediction': np.full((32, 128, 3), self.value), 'target': np.zeros((32, 128, 3)),
                'starts': np.array(study.STARTS, dtype=np.int64)}[key]
    monkeypatch.setattr(study.np, 'load', lambda path, **kwargs: Archive(path))
    admission = {'prior_continuation': previous(), 'reference_banks': banks}
    answer = study.rescore_references(admission, {'y_scale': np.array([2., 3., 4.])}, targets)
    assert answer['rescored_continuation']['family_means']['candidate'] == 2.
    assert answer['rescored_continuation']['family_means']['strong'] == 4.
    assert answer['historical_continuation'] == previous()
    assert len(answer['rows']) == 48
    targets[study.DEV_IDS[0]][0, 0, 0] = 1.
    with pytest.raises(ValueError, match='targets match'):
        study.rescore_references(admission, {'y_scale': np.ones(3)}, targets)


def opaque_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    sources = {}
    for name in study.MIN_SOURCES:
        p = tmp_path/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('opaque source '+name)
        sources[name] = study.sha(p)
    prerequisites = {}
    for name in study.REQUIRED:
        p = tmp_path/'prerequisites'/(name+'.json')
        p.parent.mkdir(exist_ok=True)
        study.write(p, {'status': 'PASS'})
        prerequisites[name] = {'path': str(p.relative_to(tmp_path)), 'sha256': study.sha(p)}
    qroot = tmp_path/'qualification'
    qroot.mkdir()
    qualified_sources = {name: study.pin(tmp_path/name) for name in sources}
    for name in sources:
        dest = qroot/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((tmp_path/name).read_bytes())
    command = ['python', '-m', 'pytest', 'research/fsm_author/tests/test_nllfr_factorial.py']
    definition = qroot/'definition.json'
    study.write(definition, {'sources': qualified_sources, 'commands': [command]})
    log = qroot/'tests.log'
    log.write_text('synthetic original success')
    qpath = tmp_path/prerequisites['producer_qualification']['path']
    study.write(qpath, {'status': 'PASS', 'definition': study.pin(definition),
        'sources_before': qualified_sources, 'sources_after': qualified_sources,
        'commands': [{'command': command, 'returncode': 0, 'log': study.pin(log)}]})
    prerequisites['producer_qualification']['sha256'] = study.sha(qpath)
    review = tmp_path/prerequisites['source_review']['path']
    study.write(review, {'status': 'PASS', 'reviewed_source_sha256': sources,
        'qualification': prerequisites['producer_qualification']})
    prerequisites['source_review']['sha256'] = study.sha(review)
    cfg = {'version': study.VERSION, 'experiment': copy.deepcopy(study.EXPERIMENT), 'source_sha256': sources,
        'prerequisites': prerequisites, 'output': 'output/study', 'process_directory': 'output/process'}
    path = tmp_path/'registration.json'
    study.write(path, cfg)
    monkeypatch.setattr(study, 'parent_admission', lambda paths: {'opaque': 'parents'})
    monkeypatch.setattr(study.np, 'load', lambda *a, **k: pytest.fail('admission decoded arrays'))
    return path, cfg


@pytest.mark.parametrize('tamper', [None, 'source', 'prerequisite', 'qualification', 'review', 'recipe', 'output'])
def test_metadata_source_qualification_input_guards_precede_arrays(tmp_path, monkeypatch, tamper):
    path, cfg = opaque_registration(tmp_path, monkeypatch)
    if tamper == 'source':
        (tmp_path/next(iter(cfg['source_sha256']))).write_text('changed')
    elif tamper == 'prerequisite':
        (tmp_path/cfg['prerequisites']['old_final']['path']).write_text('changed')
    elif tamper in ('qualification', 'review'):
        name = 'producer_qualification' if tamper == 'qualification' else 'source_review'
        own = tmp_path/cfg['prerequisites'][name]['path']
        value = study.read(own)
        value['status'] = 'FAILED'
        study.write(own, value)
        cfg['prerequisites'][name]['sha256'] = study.sha(own)
        study.write(path, cfg)
    elif tamper == 'recipe':
        cfg['experiment']['timed_requests'] = 95
        study.write(path, cfg)
    output = tmp_path/('wrong' if tamper == 'output' else cfg['output'])
    if tamper is None:
        assert study.metadata_admission(path, output)[2] == {'opaque': 'parents'}
    else:
        with pytest.raises(ValueError):
            study.metadata_admission(path, output)


@pytest.mark.parametrize('bad', ['prefix', 'incomplete_process', 'audit_disagreement'])
def test_newfit_attribution_failure_blocks_before_cache_decode(tmp_path, monkeypatch, bad):
    # Execute the actual parent-admission branch with only inherited opaque
    # authentication stubbed; prefix eligibility is owned by this evaluator.
    oldfile, newfile = tmp_path/'old.json', tmp_path/'new.json'
    oldplan = {'prerequisites': {}}
    oldinputs = {}
    newinputs = {}
    oa = {'study': str(tmp_path/'old'), 'inputs': oldinputs}
    na = {'study': str(tmp_path/'new'), 'inputs': newinputs, 'scientific_status': 'FIT_ONLY_COMPLETE',
        'results': {'budget_only_attributable': bad != 'prefix', 'prefix': {'exact': bad != 'prefix'}, 'fit_status': 'complete'}}
    paths = {k: tmp_path/k for k in study.REQUIRED}
    paths.update(old_audit=oldfile, new_audit=newfile)
    def closed(path, *args):
        if path == newfile and bad == 'audit_disagreement':
            raise ValueError('independent audit agreement required')
        return oa if path == oldfile else na
    old = SimpleNamespace(authenticate=lambda *a: (oldplan, oldinputs, {},
        {'status': 'completed', 'observed_exit_code': 0}, {'status': 'completed', 'observed_exit_code': 0}))
    newer = SimpleNamespace(authenticate=lambda *a: ({}, newinputs, {},
        {'status': 'failed' if bad == 'incomplete_process' else 'completed', 'observed_exit_code': 0}, {}, {}))
    monkeypatch.setitem(sys.modules, 'audit_fsm_author_nllfr', old)
    monkeypatch.setitem(sys.modules, 'audit_fsm_author_nllfr_budget', newer)
    monkeypatch.setattr(study, 'closed_audit', closed)
    monkeypatch.setattr(study.np, 'load', lambda *a, **k: pytest.fail('cache decoded before attribution'))
    with pytest.raises(ValueError):
        study.parent_admission(paths)


def test_closed_audit_requires_actual_process_output_and_both_source_joins(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    source = tmp_path/'source.py'
    source.write_text('opaque')
    process, registration, log = (tmp_path/n for n in ('process.json', 'registration.json', 'log'))
    for p in (process, registration, log):
        p.write_text('{}')
    audit = tmp_path/'audit.json'
    study.write(audit, {'status': 'PASS', 'agreement': True})
    sources = {'source.py': study.pin(source)}
    receipt = {'state': 'EXITED', 'observed_exit_code': 0, 'success': True, 'sources_unchanged': True,
        'inputs_unchanged': True, 'error': None, 'closure_error': None, 'audit_output': study.pin(audit),
        'fit_process': study.pin(process), 'registration': study.pin(registration), 'sources_before': sources,
        'sources_after': sources, 'inputs_before': {}, 'inputs_after': {}, 'log': study.pin(log)}
    closure = tmp_path/'closure.json'
    study.write(closure, receipt)
    assert study.closed_audit(audit, closure, 'source.py', process, registration)['status'] == 'PASS'
    receipt['audit_output']['sha256'] = '0'*64
    study.write(closure, receipt)
    with pytest.raises(ValueError):
        study.closed_audit(audit, closure, 'source.py', process, registration)


@pytest.mark.parametrize('failure', [None, 'failed', 'timeout', 'memory_limit'])
def test_original_supervisor_closes_once_preserves_failure_and_rejects_rerun(tmp_path, monkeypatch, failure):
    # No real subprocess and no model import/call in this lifecycle fixture.
    source = HERE.parent/'scripts/run_nllfr_factorial.py'
    spec = importlib.util.spec_from_file_location('isolated_factorial_supervisor', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.chdir(tmp_path)
    producer = tmp_path/'research/fsm_author/scripts/evaluate_nllfr_factorial.py'
    producer.parent.mkdir(parents=True)
    producer.write_text('opaque child')
    registration = tmp_path/'registration.json'
    registration.write_text('{}')
    for name in ('old', 'new'):
        (tmp_path/(name+'.json')).write_text('{}')
    cfg = {'output': 'study', 'process_directory': 'process', 'experiment': {'outer_timeout_seconds': 3600, 'rss_cap_bytes': 32*1024**3},
        'source_sha256': {'research/fsm_author/scripts/evaluate_nllfr_factorial.py': module.sha(producer)},
        'prerequisites': {k+'_process': {'path': k+'.json'} for k in ('old', 'new')}}
    monkeypatch.setattr(module, '__file__', str(producer.with_name('run_nllfr_factorial.py')))
    Path(module.__file__).write_text('opaque supervisor')
    monkeypatch.setattr(module, 'relative', lambda name: tmp_path/name)
    monkeypatch.setattr(module, 'metadata_admission', lambda *a: (cfg, {}, {'fixed': 'admission'}))
    spawned = []
    class Child:
        pid = 1234
        def __init__(self):
            self.done = False
        def poll(self):
            return (1 if failure == 'failed' else 0) if self.done or failure not in ('timeout', 'memory_limit') else None
        def wait(self):
            self.done = True
            return -9 if failure in ('timeout', 'memory_limit') else (1 if failure == 'failed' else 0)
    def launch(*args, **kwargs):
        spawned.append(args)
        (tmp_path/'study').mkdir()
        (tmp_path/'study'/'partial.json').write_text('{}')
        return Child()
    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=str(33*1024**2)))
    monkeypatch.setattr(module.os, 'killpg', lambda *a: None)
    ticks = iter([0., 4000., 4001.]) if failure == 'timeout' else iter([0., 1., 2., 3.])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(module.time, 'time', lambda: 100.)
    assert module.supervise(registration) == (0 if failure is None else 1)
    receipt = study.read(tmp_path/'process/process.json')
    assert receipt['status'] == (failure or 'completed') and 'partial.json' in receipt['artifacts']
    assert len(spawned) == 1
    with pytest.raises(ValueError, match='already exists'):
        module.supervise(registration)
    assert len(spawned) == 1
