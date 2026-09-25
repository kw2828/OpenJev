# SPDX-License-Identifier: GPL-3.0-or-later
"""Actual fabricated producer serialization through the complete audit body.

Inference and parent admission are explicit stubs. This tests the integration
schema, scoring, evidence inventory and accounting, not the numerical solver.
The independently qualified physical cross-parity tests cover that mathematics.
"""
import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent/'src'))
sys.path.insert(0, str(HERE.parent/'scripts'))
sys.path.insert(0, str(HERE))
import evaluate_nllfr_factorial as producer  # noqa: E402
from test_nllfr_context import model  # noqa: E402

SPEC = importlib.util.spec_from_file_location('factorial_cross_schema_audit', ROOT/'scripts/audit_fsm_author_factorial.py')
auditor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auditor)


def diagnostic(iterations):
    states = np.zeros((1, 28))
    singular = np.array([1., 1., 1.]+[0.]*25)
    detail = {'status': 'GRADIENT_TOL', 'initial_objective': 0., 'final_objective': 0.,
        'directions_considered': 1, 'accepted_steps': 0, 'trajectory_evaluations': 2,
        'jacobian_evaluations': 2, 'trial_attempts': 0, 'linear_rank': 3,
        'final_jacobian_rank': 3, 'final_jacobian_singular_values': singular.copy(),
        'trace': [{'iteration': 0, 'objective': 0., 'trials': [], 'scaled_gradient_inf': 0., 'scaled_jacobian_rank': 3}],
        'status_scope': 'bounded context solve; no optimality or forecasting-accuracy certificate'}
    return {'requests': [detail], 'linear_seed_states': states.copy(), 'solved_context_start_states': states.copy(),
        'linear_seed_diagnostics': [{'rank': 3, 'singular_values': singular.copy(), 'rcond': 1e-12,
                                    'time': 'second observed output'}],
        'policy': {'iterations': iterations, 'trials': 8, 'damping': .001, 'armijo': .0001,
                   'gradient_tol': 1e-8, 'scale_floor': 1e-8, 'rcond': 1e-12}}


def legal(start):
    return {'starts': np.array([start], dtype=np.int64), 'y_context': np.zeros((1, 100, 3)),
        'u_context': np.zeros((1, 99, 3)), 'future_u': np.zeros((1, 128, 3))}


def returned(iterations):
    return {'prediction': np.ones((1, 128, 3)), 'final_state': np.zeros((1, 28)),
        'forecast_state': np.zeros((1, 28)), 'context': diagnostic(iterations),
        'timing': {'request_ms': 4., 'copy_ms': 1., 'initializer_ms': 2., 'rollout_ms': 1.}}


def historical_maps():
    families = (*auditor.old.FAMILIES, auditor.old.BLA)
    means = {f: (.25 if f == auditor.old.CANDIDATE else 1.5 if f == auditor.old.BLA else 2.) for f in families}
    records = {f: dict.fromkeys(producer.DEV_IDS, value) for f, value in means.items()}
    seeds = {f: {str(s): means[f] for own, s in auditor.old.ORDERED if own == f} for f in auditor.old.FAMILIES}
    seeds[auditor.old.BLA] = {'None': 1.5}
    return {'status': 'DO_NOT_CONTINUE_REFERENCE_CHECK', 'reference_complete': False,
        'candidate': auditor.old.CANDIDATE, 'strongest_control': auditor.old.BLA,
        'conditions': dict.fromkeys(('five_percent_below_strongest_control', 'every_seed_below_strongest_control',
            'no_record_over_two_percent_strongest_control', 'both_amplitudes_below_strongest_control'), False),
        'passed': 0, 'total': 4, 'family_means': means, 'per_record_means': records, 'per_seed_means': seeds,
        'per_amplitude_means': {f: {'100mV': v, '200mV': v} for f, v in means.items()},
        'seed_rule': 'Paired seed for stochastic controls; common deterministic score otherwise. No new candidate selection.'}


def write_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def test_actual_serialized1536_by48_by100_completed_audit(tmp_path, monkeypatch):
    # Every file below is fabricated in tmp_path. No archive, recording or
    # trained weight path is looked up, and no inference function is executed.
    source_root, reference = tmp_path/'sources', tmp_path/'old-diagnostic'
    source_root.mkdir()
    (source_root/'fixture.py').write_text('fabricated frozen source\n')
    old_final, new_final, common = (tmp_path/name for name in ('old.npz', 'new.npz', 'normalizer.npz'))
    for path in (old_final, new_final):
        write_npz(path, **model(28))
    normalizer = {'u_mean': np.zeros(3), 'u_scale': np.ones(3), 'y_mean': np.zeros(3), 'y_scale': np.ones(3)}
    write_npz(common, **normalizer)
    for rid in producer.DEV_IDS:
        for start in producer.STARTS:
            base = reference/'evaluation'/rid/f'{start:04d}'
            inputs, value = legal(start), returned(16)
            write_npz(base.with_suffix('.inputs.npz'), **inputs)
            write_npz(base.with_suffix('.npz'), prediction=value['prediction'], target=np.zeros((1, 128, 3)),
                starts=inputs['starts'], final_state=value['final_state'], forecast_state=value['forecast_state'],
                linear_seed_states=value['context']['linear_seed_states'],
                solved_context_start_states=value['context']['solved_context_start_states'],
                **{k: inputs[k] for k in ('y_context', 'u_context', 'future_u')})
            producer.write(base.with_suffix('.json'), {'context': value['context']})

    prior, reference_banks, evaluations = historical_maps(), [], []
    reference_folder, bla = tmp_path/'controls', tmp_path/'bla'
    for family, seed in auditor.old.ORDERED:
        rows = []
        for rid in producer.DEV_IDS:
            path = reference_folder/(family+(f'-{seed}' if seed is not None else ''))/'evaluation'/(rid+'.npz')
            write_npz(path, prediction=np.full((32, 128, 3), prior['family_means'][family]),
                target=np.zeros((32, 128, 3)), starts=np.array(producer.STARTS, dtype=np.int64))
            reference_banks.append({'family': family, 'seed': seed, 'record_id': rid, 'file': producer.pin(path)})
            rows.append({'record_id': rid, 'status': 'complete'})
        evaluations.append({'family': family, 'seed': seed, 'rows': rows})
    producer.write(reference_folder/'evaluations.json', evaluations)
    for rid in producer.DEV_IDS:
        path = bla/'evaluation'/(rid+'.npz')
        write_npz(path, prediction=np.full((32, 128, 3), 1.5), target=np.zeros((32, 128, 3)),
            starts=np.array(producer.STARTS, dtype=np.int64), final_state=np.zeros((32, 28)),
            singular_values=np.ones(28), context_residual_norm=np.zeros(32),
            y_context=np.zeros((32, 100, 3)), u_context=np.zeros((32, 99, 3)), future_u=np.zeros((32, 128, 3)))
        reference_banks.append({'family': auditor.old.BLA, 'seed': None, 'record_id': rid, 'file': producer.pin(path)})
    producer.write(bla/'evaluation.json', {'rows': [{'record_id': rid, 'status': 'complete'} for rid in producer.DEV_IDS]})
    parent = {'evaluation': str(reference), 'prior_continuation': prior, 'reference_banks': reference_banks,
        'new_fit_status': 'FIT_ONLY_COMPLETE', 'new_fit_iterations': 12345}
    cfg = {'source_sha256': {'fixture.py': producer.sha(source_root/'fixture.py')}, 'prerequisites': {}}
    paths = {'old_final': old_final, 'new_final': new_final, 'common_normalizer': common}
    preflight = {'python': 'fabricated', 'executable': 'fabricated-python', 'versions': {'numpy': 'fabricated'},
        'upstream_direct_url': {'commit': 'fabricated'}, 'installed_source_matches': {'module.py': {'sha256': 'opaque'}}}
    preflight_path = tmp_path/'preflight.json'
    producer.write(preflight_path, preflight)
    paths['runtime_preflight'] = preflight_path
    identity = {'source_sha256': cfg['source_sha256'], 'prerequisites': cfg['prerequisites'],
        'python': preflight['python'], 'executable': preflight['executable'], 'versions': preflight['versions'],
        'direct_url': preflight['upstream_direct_url'], 'installed_author_sha256': {'module.py': 'opaque'},
        'thread_environment': {**producer.ENV, 'XLA_FLAGS': None}}
    monkeypatch.setattr(producer, 'ROOT', source_root)
    monkeypatch.setattr(producer, 'metadata_admission', lambda *a: (cfg, paths, parent))
    monkeypatch.setattr(producer.prior, 'identities', lambda *a: copy.deepcopy(identity))
    monkeypatch.setattr(producer.sys, 'addaudithook', lambda *a: None)
    calls = []
    def fake_request(arrays, cached, iterations):
        assert arrays['A'].shape == (28, 28) and cached['y_context'].shape == (1, 100, 3)
        assert set(cached) == set(producer.LEGAL_KEYS)
        calls.append(iterations)
        return returned(iterations)
    monkeypatch.setattr(producer, 'request', fake_request)
    output, registration = tmp_path/'study', tmp_path/'registration.json'
    output.mkdir()
    producer.write(registration, cfg)
    producer.write(output/'started.json', {'registration_sha256': producer.sha(registration), 'pid': 1234})
    producer.run(registration, output)
    assert len(calls) == 1536+4+96

    # Real independent decoding, bank/row rescoring, reference rescoring,
    # diagnostics, timing and decision checks run on the producer-written files.
    files = producer.inventory(output)
    inputs = {'files': files}
    reference_paths = {'parent_evaluations': reference_folder/'evaluations.json', 'bla_final_npz': bla/'final.npz'}
    monkeypatch.setattr(auditor, 'authenticate', lambda *a:
        (cfg, inputs, paths, {'status': 'completed'}, parent, reference_paths))
    replays = []
    def fake_independent(arrays, y, u, future, *, iterations):
        assert y.shape == (1, 100, 3) and u.shape == (1, 99, 3) and future.shape == (1, 128, 3)
        assert arrays['A'].shape == (28, 28)
        replays.append(iterations)
        value = returned(iterations)
        return value['prediction'], value['final_state'], value['forecast_state'], value['context']
    monkeypatch.setattr(auditor.replay, 'replay_request', fake_independent)
    audited = auditor.audit(output, tmp_path/'process.json', tmp_path/'freeze.json')
    assert len(replays) == 1536
    assert audited['status'] == 'PASS' and audited['agreement'] is True
    assert audited['scientific_status'] == 'REFERENCE_COMPLETE'
    assert audited['counts'] == {'request_replays': 1536, 'reference_banks_rescored': 312, 'timing_replays': 0,
        'optimizer_calls': 0, 'raw_archive_decodes': 0, 'new_recording_decodes': 0,
        'record_slots': 48, 'complete_request_banks': 1536, 'failed_request_replays': 0, 'warmups': 4, 'timing_slots': 96}
    assert audited['results']['continuation']['passed'] == 4
    assert set(producer.inventory(output)) == set(files)
