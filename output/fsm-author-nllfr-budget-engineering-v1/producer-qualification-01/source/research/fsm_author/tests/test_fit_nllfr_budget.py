# SPDX-License-Identifier: GPL-3.0-or-later
"""Opaque admission/process fixtures and one small synthetic vendor prefix check."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parents[2]/'scripts'))
spec = importlib.util.spec_from_file_location('fit_nllfr_budget', SCRIPTS/'fit_nllfr_budget.py')
fit = importlib.util.module_from_spec(spec)
sys.modules['fit_nllfr_budget'] = fit
spec.loader.exec_module(fit)

spec = importlib.util.spec_from_file_location('budget_supervisor', SCRIPTS/'run_nllfr_budget_study.py')
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if not isinstance(value, str) else value)
    return path


def fixture(tmp_path, monkeypatch):
    import audit_fsm_author_nllfr as parent
    monkeypatch.setattr(fit, 'ROOT', tmp_path)
    monkeypatch.setattr(supervisor, 'ROOT', tmp_path)
    for name in fit.MIN_SOURCES:
        put(tmp_path/name, '# fabricated source')
    monkeypatch.setattr(supervisor, '__file__', str(tmp_path/'research/fsm_author/scripts/run_nllfr_budget_study.py'))
    sources = {name: fit.sha(tmp_path/name) for name in fit.MIN_SOURCES}
    prerequisites = {}
    def evidence(key, value, name=None):
        path = put(tmp_path/(name or 'evidence/'+key+'.json'), value)
        prerequisites[key] = {'path': str(path.relative_to(tmp_path)), 'sha256': fit.sha(path)}
        return path
    evidence('runtime_preflight', {'status': 'PASS'})
    parent_folder = tmp_path/'parent-fit'
    for key, name in fit.PARENT_FILES.items():
        evidence(key, 'opaque invalid archive: no decoding', 'parent-fit/'+name)
    put(parent_folder/'fit.json', {'iterations': 10000})
    reg = evidence('parent_registration', {'original': True})
    process = evidence('parent_process', {'status': 'completed'})
    evaluation = evidence('parent_evaluation_process', {'status': 'completed'})
    current = {k: fit.pin(tmp_path/k) for k in sources}
    for name in sources:
        put(tmp_path/'source'/name, '# fabricated source')
    inputs = {'registration': fit.pin(reg), 'process': fit.pin(process), 'evaluation_process': fit.pin(evaluation),
              'source': current['scripts/audit_fsm_author_nllfr.py'],
              'files': {name: {k: fit.pin(parent_folder/name)[k] for k in ('sha256', 'bytes')}
                        for name in (*fit.PARENT_FILES.values(), 'fit.json')}}
    audit = evidence('parent_audit', {'status': 'PASS', 'agreement': True, 'study': str(parent_folder),
        'inputs': inputs, 'results': {'fit_status': 'iteration_cap_reached'}})
    log = put(tmp_path/'parent-audit.log', 'PASS')
    evidence('parent_audit_process', {'state': 'EXITED', 'observed_exit_code': 0, 'success': True,
        'sources_unchanged': True, 'inputs_unchanged': True, 'error': None, 'closure_error': None,
        'audit_output': fit.pin(audit), 'fit_process': fit.pin(process), 'evaluation_process': fit.pin(evaluation),
        'sources_before': {'scripts/audit_fsm_author_nllfr.py': inputs['source']},
        'sources_after': {'scripts/audit_fsm_author_nllfr.py': inputs['source']},
        'inputs_before': {'process': fit.pin(process)}, 'inputs_after': {'process': fit.pin(process)}, 'log': fit.pin(log)})
    terminal = {'status': 'completed', 'observed_exit_code': 0}
    monkeypatch.setattr(parent, 'authenticate', lambda *args: ({}, inputs, {}, terminal, terminal))
    command = ['python', '-m', 'pytest', 'research/fsm_author/tests/test_fit_nllfr_budget.py']
    definition = put(tmp_path/'definition.json', {'sources': current, 'commands': [command]})
    log = put(tmp_path/'qualification.log', 'PASS')
    evidence('producer_qualification', {'status': 'PASS', 'definition': fit.pin(definition),
        'sources_before': current, 'sources_after': current,
        'commands': [{'command': command, 'returncode': 0, 'log': fit.pin(log)}]})
    evidence('source_review', {'status': 'PASS', 'qualification': prerequisites['producer_qualification'],
                             'reviewed_source_sha256': sources})
    cfg = {'version': fit.VERSION, 'experiment': fit.EXPERIMENT.copy(), 'source_sha256': sources,
           'prerequisites': prerequisites, 'output': 'output/budget', 'process_directory': 'output/process'}
    registration = put(tmp_path/'registration.json', cfg)
    return cfg, registration


def test_exact_admission_is_opaque_and_only_budget_changes(tmp_path, monkeypatch):
    cfg, reg = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('metadata must precede decoding'))
    assert fit.metadata_admission(reg) == cfg
    delta = {k for k in fit.EXPERIMENT if fit.EXPERIMENT[k] != fit.base.EXPERIMENT[k]}
    assert delta == {'max_iter'} and fit.EXPERIMENT['max_iter'] == 100000


@pytest.mark.parametrize('change', ['source', 'payload', 'final_as_initial', 'recipe', 'dev_phase',
                                   'parent_audit', 'live_process', 'qualification', 'review', 'log'])
def test_admission_tampering_fails_before_arrays(tmp_path, monkeypatch, change):
    cfg, reg = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('no array reads'))
    def replace(key, mutate):
        path = fit.prerequisite(cfg, key)
        value = fit.read(path)
        mutate(value)
        put(path, value)
        cfg['prerequisites'][key]['sha256'] = fit.sha(path)
    if change == 'source':
        put(tmp_path/next(iter(cfg['source_sha256'])), '# changed')
    elif change == 'payload':
        put(fit.prerequisite(cfg, 'parent_fit_data'), 'changed')
    elif change == 'final_as_initial':
        original = fit.prerequisite(cfg, 'parent_initial_npz')
        final = original.with_name('final.npz')
        final.write_bytes(original.read_bytes())
        cfg['prerequisites']['parent_initial_npz']['path'] = str(final.relative_to(tmp_path))
    elif change == 'recipe':
        cfg['experiment']['rtol'] = .002
    elif change == 'dev_phase':
        cfg['evaluation_output'] = 'output/no'
    elif change == 'parent_audit':
        replace('parent_audit', lambda x: x.update(agreement=False))
    elif change == 'live_process':
        replace('parent_audit_process', lambda x: x.update(state='running'))
    elif change == 'qualification':
        replace('producer_qualification', lambda x: x.update(status='FAIL'))
    elif change == 'review':
        replace('source_review', lambda x: x.update(reviewed_source_sha256={}))
    else:
        put(tmp_path/'qualification.log', 'changed')
    put(reg, cfg)
    with pytest.raises(ValueError):
        fit.metadata_admission(reg)


def test_prefix_exact_mismatch_and_insufficient_are_distinct():
    old = np.array([4., 4., 3., 2.])
    assert fit.prefix_comparison(np.r_[old, 1.], old, np, required=4) == {
        'required': 4, 'compared': 4, 'exact': True, 'first_difference': None, 'max_absolute_difference': 0.}
    mismatch = fit.prefix_comparison(np.array([4., 4., 3.25, 2., 1.]), old, np, required=4)
    assert mismatch['first_difference'] == 2 and mismatch['max_absolute_difference'] == .25
    assert fit.fit_status(True, mismatch) == ('prefix_mismatch', 'complete')
    short = fit.prefix_comparison(old[:2], old, np, required=4)
    assert not short['exact'] and short['compared'] == 2 and short['first_difference'] is None
    assert fit.fit_status(False, {'exact': True}) == ('iteration_cap_reached', 'iteration_cap_reached')
    assert fit.fit_status(True, {'exact': True}) == ('complete', 'complete')


def test_nonfinite_prefix_is_retained_not_json_nan():
    result = fit.prefix_comparison(np.array([4., np.nan]), np.array([4., 2.]), np, required=2)
    assert result == {'required': 2, 'compared': 2, 'exact': False,
                      'first_difference': 1, 'max_absolute_difference': None}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('outcome', ['completed', 'failed', 'timeout', 'memory_limit'])
def test_original_process_preserves_partial_results_and_no_retry(tmp_path, monkeypatch, outcome):
    cfg, reg = fixture(tmp_path, monkeypatch)
    if outcome == 'timeout':
        cfg['experiment']['outer_timeout_seconds'] = 0
    if outcome == 'memory_limit':
        cfg['experiment']['rss_cap_bytes'] = 1
    monkeypatch.setattr(supervisor, 'metadata_admission', lambda *a: cfg)
    monkeypatch.chdir(tmp_path)
    calls, killed = [], []
    class Child:
        pid = 123456
        polls = 0
        code = None
        def poll(self):
            self.polls += 1
            if outcome in ('completed', 'failed') and self.polls >= 3:
                self.code = 0 if outcome == 'completed' else 7
            return self.code
        def wait(self):
            return self.code
    child = Child()
    def launch(command, **kwargs):
        calls.append(command)
        assert kwargs['env'] == {**supervisor.os.environ, **fit.ENV}
        assert kwargs['start_new_session']
        put(tmp_path/cfg['output']/'partial.json', {'retained': True})
        return child
    def kill(pid, sig):
        killed.append(pid)
        child.code = -9
    monkeypatch.setattr(supervisor.subprocess, 'Popen', launch)
    monkeypatch.setattr(supervisor.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='2'))
    monkeypatch.setattr(supervisor.os, 'killpg', kill)
    monkeypatch.setattr(supervisor.time, 'sleep', lambda *a: None)
    assert supervisor.supervise(reg) == (0 if outcome == 'completed' else 1)
    receipt = fit.read(tmp_path/cfg['process_directory']/'process.json')
    assert receipt['status'] == outcome and receipt['end_identity_matches']
    assert receipt['parent_fit_process'] == fit.pin(fit.prerequisite(cfg, 'parent_process'))
    assert 'partial.json' in receipt['artifacts'] and len(calls) == 1
    assert bool(killed) == (outcome in ('timeout', 'memory_limit'))
    with pytest.raises(ValueError, match='already exists'):
        supervisor.supervise(reg)
    assert len(calls) == 1


def test_overlapping_evidence_rejected_before_launch(tmp_path, monkeypatch):
    cfg, reg = fixture(tmp_path, monkeypatch)
    cfg['output'] = 'parent-fit'
    monkeypatch.setattr(supervisor, 'metadata_admission', lambda *a: cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(supervisor.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='overwrites input'):
        supervisor.supervise(reg)


def test_small_fabricated_vendor_budget_prefix(tmp_path):
    """Unchanged vendor routine, two fresh optimizers, exact2-step common prefix.

    Small order2/two-width4 fixture checks the max_iter path and serialization.
    The existing full7473-parameter qualification supplies the full-size evidence.
    """
    import freq_statespace as fss
    import jax
    import optimistix as optx

    from openjev_fsm_author import nllfr
    from openjev_fsm_author.benchmark import export_model

    assert all(fit.os.environ.get(k) == v for k, v in fit.ENV.items())
    assert jax.config.jax_enable_x64 and all(d.platform == 'cpu' for d in jax.devices())
    n = 64
    A = np.diag([.35, .7])
    B = np.array([[1., .3, -.2], [.2, -.4, .7]])
    C = np.array([[1., .2], [-.3, .5], [.2, -.6]])
    D = np.diag([.1, .2, .3])
    U = np.zeros((n//2+1, 3, 3), dtype=np.complex128)
    for k in range(1, n//2):
        U[k] = np.exp(2j*np.pi*k*np.arange(1, 4)[:, None]/13)*np.exp(-2j*np.pi*np.arange(3)[:, None]*np.arange(3)[None]/3)
    z = np.exp(2j*np.pi*np.arange(n//2+1)/n)
    response = C[None]@np.linalg.solve(z[:, None, None]*np.eye(2)-A, np.broadcast_to(B, (len(z), 2, 3)))
    Y = (response+D+.05*np.eye(3))@U
    u, y = [np.repeat(np.fft.irfft(v, n=n, axis=0)[..., None], 2, axis=3) for v in (U, Y)]
    data = fss.create_data_object(u, y, np.arange(1, n//2), 6400.)
    bla = fss.ModelBLA(A=A, B_u=B*np.asarray(data.norm.u_std)[None],
        C_y=C/np.asarray(data.norm.y_std)[:, None],
        D_yu=D*np.asarray(data.norm.u_std)[None]/np.asarray(data.norm.y_std)[:, None], ts=1/6400, norm=data.norm)
    net = fss.static.NeuralNetwork(nz=2, nw=2, layers=2, neurons_per_layer=4, activation=jax.nn.relu, seed=42, bias=True)
    initial = fss.nonlin.connect(bla, net, sigma=1e-4)
    fss.save_model(initial, tmp_path/'initial.zip')
    reloaded = fss.load_model(tmp_path/'initial.zip')
    assert type(initial.ts) is type(initial._bla.ts) is type(reloaded.ts) is type(reloaded._bla.ts) is float
    for key, value in nllfr.export(initial).items():
        np.testing.assert_array_equal(nllfr.export(reloaded)[key], value)
    for key, value in export_model(initial._bla).items():
        np.testing.assert_array_equal(export_model(reloaded._bla)[key], value)
    results = []
    for limit in (2, 4):
        start = initial if limit == 2 else fss.load_model(tmp_path/'initial.zip')
        model, trace = fss.nonlin.optimize(start, data,
            solver=optx.BFGS(rtol=.001, atol=.00001), freq_weighting=False, max_iter=limit,
            print_every=-1, return_solve_details=True, offset=None, device='cpu')
        fss.save_model(model, tmp_path/f'final-{limit}.zip')
        np.savez(tmp_path/f'trace-{limit}.npz', loss_history=trace.loss_history, iter_count=trace.iter_count)
        assert trace.iter_count == limit and np.isfinite(trace.loss_history).all()
        nllfr.export(model)
        results.append(np.asarray(trace.loss_history))
    assert fit.prefix_comparison(results[1], results[0], np, required=2)['exact']
