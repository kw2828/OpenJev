"""Fixed twelve-slot common-root diagnostic; no RNG during native execution."""
import copy
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from prepare_inputs import BASE, NAMES, ROOT, prior_records, protected_sources, root_record
from prepare_inputs import OUT as INPUTS

from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.reacher_common_root_audit import audit_slot
from openjev.research.reacher_common_root_branches import build_union, run_branches
from openjev.research.reacher_geometry_physics import PhysicsGeometryCEM
from openjev.research.reacher_search_protocol import load_inputs, save_npz
from openjev.research.reacher_tracking_control import GainPlanningBank
from openjev.research.reacher_tracking_dynamics import nominal_model
from openjev.research.reacher_tracking_rollout import check_deadline, save_decision, sha, write

OUT = BASE/'attempt-01'
NEW = ['src/openjev/research/'+name+'.py' for name in ('reacher_common_root_branches', 'reacher_common_root_audit')]
NEW += [(BASE/name).relative_to(ROOT).as_posix() for name in ('protocol.json', 'prepare_inputs.py', 'run_diagnostic.py', 'report_results.py')]
NEW += ['tests/test_'+name+'.py' for name in ('reacher_common_root_branches', 'reacher_common_root_audit', 'reacher_common_root_report')]
NEW += [(BASE/'report-arithmetic.json').relative_to(ROOT).as_posix()]


def array_mapping(value):
    return dict(zip(NAMES, (value.initial, value.random_extra, *value.cem), strict=True))


def prefix(value):
    arrays = [x[:, :, :4] for x in array_mapping(value).values()]
    return SearchInputs(arrays[0], arrays[1], tuple(arrays[2:]))


def prepared(protocol):
    complete = json.loads((INPUTS/'completed.json').read_text())
    assert complete['status'] == 'completed' and not (INPUTS/'failed.json').exists()
    members = {p.name for p in INPUTS.iterdir() if p.is_file()}
    assert members == set(complete['files']) | {'completed.json'}
    for name, digest in complete['files'].items():
        assert not (INPUTS/name).is_symlink() and sha(INPUTS/name) == digest
    started = json.loads((INPUTS/'started.json').read_text())
    assert started['protocol_sha256'] == sha(BASE/'protocol.json')
    assert started['preparer_sha256'] == sha(BASE/'prepare_inputs.py')
    prior, records = prior_records(protocol)
    roots = json.loads((INPUTS/'roots.json').read_text())
    expected = {f'{role}--{t:03d}': root_record(role, t, records[role])
                for role in protocol['roles'] for t in protocol['root_steps']}
    assert roots == expected
    arrays = {}
    for t in protocol['root_steps']:
        binding = complete['original_inputs'][str(t)]
        for record in records.values():
            assert record['started']['inputs_by_step'][t] == binding
        stem = Path(binding['stem'])
        for suffix in ('npz', 'json'):
            assert sha(stem.with_suffix('.'+suffix)) == binding[suffix+'_sha256']
        old = load_inputs(stem)
        a = load_inputs(INPUTS/f'{t:03d}-A'); b = load_inputs(INPUTS/f'{t:03d}-B')
        for name, value in array_mapping(old).items():
            assert np.array_equal(value, array_mapping(a)[name][:, :, :4])
        with np.load(INPUTS/f'{t:03d}-noise.npz', allow_pickle=False) as saved:
            assert set(saved.files) == {'noise'}
            noise = saved['noise'].copy()
        assert noise.shape == (4, 24, 2) and noise.dtype == np.float64 and np.isfinite(noise).all()
        arrays[t] = (a, b, noise)
    return prior, roots, arrays


def execute_slot(folder, model, root, a, b, noise, deadline):
    started_at = time.monotonic(); folder.mkdir(parents=True, exist_ok=False)
    active = None; stage = 'root'; outcomes = {}; initial = None
    try:
        write(folder/'root.json', root)
        q = np.array(root['root']['qpos'], np.float64)[None]
        v = np.array(root['root']['qvel'], np.float64)[None]
        packet = np.array(root['public_packet'], np.float32); target = packet[None, 4:6]
        for assumption, gain in [('nominal', 1.), ('actual', root['true_gain'])]:
            for label, inputs, horizon in [('short_a', prefix(a), 12), ('short_b', prefix(b), 12), ('long_a', a, 24)]:
                stage = f'{assumption}--{label}'; check_deadline(deadline)
                active = GainPlanningBank(model, allowed_gains=np.array([gain]), steps=200,
                    noise_std=.05, planning_horizon=horizon, action_block=3)
                result, journal, snapshot = active.plan(q, v, target, gain, inputs, step=root['step'], deadline=deadline)
                dest = folder/'searches'/stage; (dest/'decisions').mkdir(parents=True)
                save_decision(dest, {'step': root['step'], 'arm': assumption, 'root_qpos': q[0],
                    'root_qvel': v[0], 'public_packet': packet, 'public_qpos': q[0], 'public_qvel': v[0],
                    'action': result.selected_actions[0], 'planning_gain': gain, 'search': result,
                    'journal': journal, 'bank_snapshot': snapshot}, inputs)
                outcomes[(assumption, label)] = {'sequence': result.selected_sequences[0].copy(),
                    'score': float(result.scores[0, result.selected_ids[0]])}
                if label == 'long_a':
                    bank = result.sequences[0, :64].copy()
                    if initial is None:
                        initial = bank
                    else:
                        assert np.array_equal(initial, bank)
        selected = []
        for assumption in ('nominal', 'actual'):
            first = outcomes[(assumption, 'short_a')]; second = outcomes[(assumption, 'short_b')]
            best = first if first['score'] >= second['score'] else second
            selected.extend([first['sequence'], best['sequence'], outcomes[(assumption, 'long_a')]['sequence']])
        union = build_union(initial, selected); save_npz(folder/'union.npz', **union)
        stage = 'cross'; (folder/'cross').mkdir()
        for assumption, gain in [('nominal', 1.), ('actual', root['true_gain'])]:
            check_deadline(deadline); specific = copy.copy(model); specific.actuator_gear[:, 0] = 200*gain
            active = PhysicsGeometryCEM(specific, noise_std=.05, steps=200, planning_horizon=24, action_block=3)
            _, journal = active.score_bank(q, v, target, union['unique'][None], step=root['step'], deadline=deadline)
            record = journal['banks'][0]; data = record['arrays']
            reward = data['geometry_reward'][0]; summed = np.zeros(len(reward), np.float32)
            prefix_score = None
            for t in range(24):
                summed += np.clip(reward[:, t], -2.5, 0.)
                if t == 11:
                    prefix_score = summed.copy()
            save_npz(folder/'cross'/f'{assumption}.npz', sequences=union['unique'],
                predicted_angles=data['predicted_angles'][0], raw_rewards=reward,
                scores_12=prefix_score, scores_24=summed, qpos=data['qpos'][0], qvel=data['qvel'][0])
            write(folder/'cross'/f'{assumption}.json', {'configuration': journal['configuration'],
                                                       'journalmetadata': record['metadata']})
        stage = 'branches'; active = None
        run_branches(model, root=root['root'], true_gain=root['true_gain'], sequences=union['unique'],
                     noise=noise, out=folder/'branches', deadline=deadline)
        check_deadline(deadline); k = len(union['unique'])
        files = {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        result = {'status': 'completed', 'wall_seconds': time.monotonic()-started_at,
                  'work': {'search_candidate': 24576, 'selected_advance': 6,
                           'cross_score': 2*k*24, 'native_branch': 4*k*24},
                  'files': files}
        write(folder/'completed.json', result); check_deadline(deadline)
        return result
    except BaseException as error:
        if (folder/'completed.json').exists():
            try:
                (folder/'completed.json').rename(folder/'partial-completed.json')
            except BaseException as secondary:  # noqa: BLE001 - preserve original failure.
                error.add_note('Completion demotion failed: '+repr(secondary))
        if active is not None:
            try:
                write(folder/'partial-planner.json', active.snapshot())
            except BaseException as capture:  # noqa: BLE001 - preserve original failure.
                error.add_note('Planner snapshot failure: '+repr(capture))
        try:
            write(folder/'failed.json', {'status': 'failed', 'stage': stage, 'error': repr(error),
                                        'wall_seconds': time.monotonic()-started_at})
        except BaseException as capture:  # noqa: BLE001 - preserve original failure.
            error.add_note('Failure receipt error: '+repr(capture))
        raise


def main():
    OUT.mkdir(exist_ok=False); started = time.monotonic(); execution_valid = False; stage = 'setup'; rows = []; audits = []
    try:
        write(OUT/'request.json', {'status': 'requested', 'driver_sha256': sha(Path(__file__)),
                                   'protocol_sha256': sha(BASE/'protocol.json')})
        torch.set_num_threads(1)
        protocol = json.loads((BASE/'protocol.json').read_text())
        prior, roots, arrays = prepared(protocol); protected = protected_sources()
        arithmetic = json.loads((BASE/'report-arithmetic.json').read_text())
        assert arithmetic['preparation_completed_sha256'] == sha(INPUTS/'completed.json')
        sources = {**prior['source_sha256'], **{p: sha(ROOT/p) for p in NEW}}
        deadline = started+protocol['execution_cap_seconds']; check_deadline(deadline)
        write(OUT/'started.json', {'status': 'started', 'protocol': protocol, 'report_arithmetic': arithmetic, 'source_sha256': sources,
            'protected_sources': protected, 'inputs_completed_sha256': sha(INPUTS/'completed.json'),
            'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
                        'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'torch', 'mujoco', 'gymnasium')},
                        'torch_threads': torch.get_num_threads()}})
        for name, digest in sources.items():
            path = OUT/'source-snapshot'/name; path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f:
                f.write((ROOT/name).read_bytes())
            assert sha(path) == digest
            check_deadline(deadline)
        model = nominal_model(); deadline = started+protocol['execution_cap_seconds']; stage = 'execution'
        for name, root in roots.items():
            a, b, noise = arrays[root['step']]
            result = execute_slot(OUT/'slots'/name, model, root, a, b, noise, deadline)
            rows.append({'name': name, 'completed_sha256': sha(OUT/'slots'/name/'completed.json'),
                         'wall_seconds': result['wall_seconds'], 'work': result['work']})
            print(json.dumps({'phase': 'slot_complete', 'name': name, 'seconds': result['wall_seconds']}), flush=True)
        work = {k: sum(row['work'][k] for row in rows) for k in rows[0]['work']}
        assert work['search_candidate'] == 294912 and work['selected_advance'] == 72
        assert sum(work.values()) <= protocol['maximum_execution_transitions']['total']
        check_deadline(deadline)
        write(OUT/'execution-completed.json', {'status': 'completed', 'rows': rows, 'work': work,
                                               'wall_seconds': time.monotonic()-started})
        check_deadline(deadline); execution_valid = True
        stage = 'audit'; audit_start = time.monotonic(); deadline = audit_start+protocol['audit_cap_seconds']
        _, expected, arrays = prepared(protocol)
        for row in rows:
            name = row['name']; root = expected[name]; a, b, noise = arrays[root['step']]
            result = audit_slot(OUT/'slots'/name, nominal_model=model, expected_root=root,
                inputs_a=array_mapping(a), inputs_b=array_mapping(b), expected_noise=noise, deadline=deadline)
            write(OUT/f'audit-{name}.json', result)
            audits.append({'name': name, 'sha256': sha(OUT/f'audit-{name}.json')})
            print(json.dumps({'phase': 'audit_complete', 'name': name}), flush=True)
        protected_sources(); prepared(protocol)
        assert all(sha(ROOT/name) == digest for name, digest in sources.items())
        check_deadline(deadline)
        write(OUT/'completed.json', {'status': 'completed', 'scientific_gate': None, 'rows': rows, 'audits': audits,
            'source_sha256': sources, 'execution_completed_sha256': sha(OUT/'execution-completed.json'),
            'work': work, 'audit_wall_seconds': time.monotonic()-audit_start,
            'total_wall_seconds': time.monotonic()-started})
        check_deadline(deadline)
    except BaseException as error:
        for name in ('completed.json', 'execution-completed.json'):
            if name == 'execution-completed.json' and execution_valid:
                continue
            if (OUT/name).exists():
                try:
                    (OUT/name).rename(OUT/('partial-'+name))
                except BaseException as secondary:  # noqa: BLE001 - preserve original failure.
                    error.add_note('Completion demotion failed: '+repr(secondary))
        try:
            write(OUT/'failed.json', {'status': 'failed', 'stage': stage, 'error': repr(error),
                'rows': rows, 'audits': audits, 'wall_seconds': time.monotonic()-started, 'automatic_retry': False})
        except BaseException as secondary:  # noqa: BLE001 - preserve original failure.
            error.add_note('Failure receipt failed: '+repr(secondary))
        raise


if __name__ == '__main__':
    main()
