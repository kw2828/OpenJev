"""Fabricated study contracts. No sampled scientific fields."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def study():
    spec = importlib.util.spec_from_file_location('measurement_study_test', ROOT/'scripts/measurement_study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(study):
    result = []
    for p in study.CONFIG['populations']:
        for c in range(3):
            for m in study.METHODS:
                regret = 0. if m in ('spectral118', 'full') else .01
                values = dict.fromkeys(study.METRICS, 0.)
                values.update(regret=regret, nll=0., always_defer_regret=.05)
                result.append({'population': p, 'cohort': c, 'method': m, 'metrics': values})
    return result


def intervals():
    return {'long': {'ci_high': -.001}, 'long_shift': {'ci_high': -.001}}


def test_all_26_conditions_and_exact_thresholds(study):
    values = rows(study)
    checks = study.conditions(values, intervals())
    assert len(checks) == 26 and all(c['pass'] for c in checks)
    assert len({c['name'] for c in checks}) == 26
    modified = next(r for r in values if (r['population'], r['cohort'], r['method']) == ('base', 0, 'spectral118'))
    modified['metrics']['regret'] = np.nextafter(1e-8, np.inf)
    assert not next(c for c in study.conditions(values, intervals()) if c['name'] == 'base/0/exact-prefix')['pass']
    zero = intervals(); zero['long']['ci_high'] = 0.
    assert not next(c for c in study.conditions(rows(study), zero) if c['name'] == 'long/paired-upper')['pass']


@pytest.mark.parametrize('mutation', ('missing', 'duplicate', 'nan', 'interval'))
def test_incomplete_gate_is_rejected(study, mutation):
    values, bounds = rows(study), intervals()
    if mutation == 'missing':
        values.pop()
    elif mutation == 'duplicate':
        values[-1] = values[0]
    elif mutation == 'nan':
        values[0]['metrics']['nll'] = float('nan')
    else:
        bounds.pop('long')
    with pytest.raises(ValueError):
        study.conditions(values, bounds)


def test_metrics_and_pairing_use_contexts_not_paths(study):
    prediction = {'mean': np.zeros((2, 2, 4)), 'variance': np.ones((2, 2, 4)),
                  'risk': np.full((2, 2, 4), .3)}
    reference = {'risk': np.broadcast_to([0., .2, .3, .4], (2, 2, 4))}
    result = study.metrics(prediction, reference, np.zeros((2, 2, 4)))
    assert result['regret'] == pytest.approx(.18)
    assert result['nll'] == pytest.approx(.5*np.log(2*np.pi))
    assert result['brier'] == pytest.approx(.09)
    assert result['defer'] == result['coverage90'] == 1.
    assert study.field_regrets(prediction, reference).shape == (2,)
    tied = study.costs(np.array([[[.18, .18, .3, .4]]]))
    assert tied.argmin(-1).item() == 0


def test_bootstrap_fixed_constant_and_independent_reconstruction(study):
    values = np.linspace(-.03, -.005, 384)
    result = study.bootstrap(values, 12)
    rng = np.random.default_rng(12)
    means = np.array([values[rng.integers(384, size=384)].mean() for _ in range(1000)])
    np.testing.assert_allclose([result['ci_low'], result['ci_high']], np.percentile(means, [2.5, 97.5]))
    assert result['context_differences'] == values.tolist()
    assert result['seed'] == 12 and result['repetitions'] == 1000
    constant = study.bootstrap(np.full(384, -.01), 99)
    assert constant['ci_low'] == pytest.approx(-.01) == constant['ci_high']
    with pytest.raises(ValueError):
        study.bootstrap(values[:-1], 12)


def test_public_stream_arity_and_duplicate_guard(study, monkeypatch):
    import inspect
    assert tuple(inspect.signature(study.retain).parameters) == ('x', 'y', 'method')
    def forbidden(*args, **kwargs):
        raise AssertionError('must reject before allocation')
    monkeypatch.setattr(study.memory, 'initial_packed', forbidden)
    with pytest.raises(ValueError, match='unique'):
        study.retain(np.zeros((1, 2, 2)), np.zeros((1, 2)), 'coverage118')


@pytest.mark.parametrize(('method', 'size'), [('spectral118', 1022), ('dct118', 1022),
    ('bins118', 1022), ('coverage118', 1021), ('recent98', 1020), ('coverage98', 1020)])
def test_all_methods_evaluate_serialize_and_charge_slot_dataclasses(study, method, size):
    axis = np.arange(-2., 2.01, .25)
    grid = np.array([(x, y) for x in axis for y in axis], np.float64)
    ids = (np.arange(120)*37) % 289
    x = grid[ids][None]
    public = {'x': x, 'y': np.sin(x[..., 0])+.2*np.cos(x[..., 1]),
              'paths': np.broadcast_to(grid[np.arange(16)].reshape(1, 1, 4, 4, 2), (1, 1, 4, 4, 2)).copy()}
    prediction, state = study.evaluate(public, method)
    assert prediction['risk'].shape == (1, 1, 4)
    assert np.isfinite(prediction['mean']).all() and (prediction['variance'] > 0).all()
    assert study.logical_bytes(state, public) == size
    arrays = study.state_arrays(state)
    assert arrays['step'].tolist() == [120]
    if method in study.HYBRID:
        assert arrays['kind'].dtype == np.uint8
    assert all(isinstance(a, np.ndarray) for a in arrays.values())


def test_registration_is_exclusive_and_source_pins_enforced(study, tmp_path, monkeypatch):
    root = tmp_path/'repo'; root.mkdir()
    (root/'a.py').write_text('source')
    parent = root/'research/retention-results/summary.json'; parent.parent.mkdir(parents=True)
    parent.write_text('{}')
    monkeypatch.setattr(study, 'ROOT', root)
    monkeypatch.setattr(study, 'SOURCES', ('a.py',))
    out = tmp_path/'registration'; study.register(out); study.verify(out)
    with pytest.raises(FileExistsError):
        study.register(out)
    (root/'a.py').write_text('changed')
    with pytest.raises(ValueError, match='frozen source'):
        study.verify(out)


def test_failure_keeps_original_exception_and_evidence(study, tmp_path, monkeypatch):
    out = tmp_path/'failure'; out.mkdir()
    (out/'registration.json').write_text('{}')
    monkeypatch.setattr(study, 'verify', lambda _: None)
    stop = RuntimeError('fabricated pre-generation failure')
    def fail(*args, **kwargs):
        raise stop
    monkeypatch.setattr(study.data_api, 'generate', fail)
    with pytest.raises(RuntimeError) as caught:
        study.run(out)
    assert caught.value is stop
    status = json.loads((out/'run-status.json').read_text())
    assert status['state'] == 'FAILED' and status['error_type'] == 'RuntimeError'
    manifest = json.loads((out/'manifest.json').read_text())['files']
    assert {'started.json', 'partial.json', 'run-status.json', 'registration.json'} <= set(manifest)
    assert not list((out/'data').iterdir())
    with pytest.raises(FileExistsError):
        study.run(out)


def test_frozen_design_has_no_training_and_complete_rosters(study):
    config = study.CONFIG
    assert config['namespace'] == 553260924
    assert config['cohorts']*config['evaluation_contexts']*len(config['populations']) == 1536
    assert len(config['methods'])*12 == 84
    assert (len(config['methods'])-1)*12 == 72
    assert len(config['methods'])*len(config['populations']) == 28
    assert len(study.SOURCES) == 8
    assert 'train' not in config and 'learned' not in config
