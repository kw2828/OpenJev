"""Metadata-only readback of the preserved fixed-quota collection failure.

Reads only the frozen plan, failed worker receipt, completed parent terminal,
and episode JSONL. No trace replay, arrays, model, simulator or producer import.
"""
import collections
import hashlib
import json
import math
import pathlib
import resource
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent
BASE = ROOT / 'output/otto-coverage-v1'
PINS = {
    'collection-plan-01.json': '75a8a2c8e16e31c68e1dfc76be84911ba52038497b5198decbe1ae2329576004',
    'collection-process-01.terminal.json': '24331b5858f8727f348a35225d1a879e03ad2c1f2772db882fe1793412664fc5',
    'collection-01/failed.json': '2f9ec0c693dd18cbd2a326a311609ff2aed2ce0fa9ff6fc44d3d753c6af29485',
    'collection-01/collection-episodes.jsonl': '64abff3608ef7722f7bec08ca43a594151e7752f1022a51f04e06edc71e284d1',
}
SCOPE = ('Metadata-only quota-failure verification. Episode membership, pre-action row counts, '
         'frozen quota arithmetic and worker/parent failure joins are recomputed. Public state, '
         'checkpoint readouts, native dynamics, RNG and detailed work journal are not replayed. '
         'The worker call ledger is a recorded witness, reconciled to episode counts. No '
         'completed-collection audit, array decoding, fitting, target generation or simulator call.')


def require(value, message):
    if not value:
        raise ValueError(message)


def desc(path):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'regular input')
    payload = path.read_bytes()
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


def write(name, value):
    with (OUT/name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    started = time.monotonic()
    inputs = {name: desc(BASE/name) for name in PINS}
    require(all(inputs[name]['sha256'] == pin for name, pin in PINS.items()), 'fixed input identities')
    plan = json.loads((BASE/'collection-plan-01.json').read_text())
    worker = json.loads((BASE/'collection-01/failed.json').read_text())
    terminal = json.loads((BASE/'collection-process-01.terminal.json').read_text())
    require(plan['status'] == 'frozen_before_execution' and plan['version'] == worker['version'] == 'otto-coverage-collection-v1', 'frozen collection identity')
    require(worker['status'] == 'failed' and worker['plan_sha256'] == PINS['collection-plan-01.json']
            and worker['sources'] == plan['sources'] and worker['inputs'] == plan['inputs']
            and worker['limits'] == plan['limits'], 'worker frozen metadata joins')
    require(worker['error'] == "ValueError('insufficient unique candidates for fixed cell quota')"
            and 'self.prepare_mixture()' in worker['traceback'] and 'self.reservoirs[key].finish()' in worker['traceback'], 'declared fixed-quota failure location')
    require(terminal['status'] == 'failed' and terminal['returncode'] == 1 and terminal['timed_out'] is False
            and terminal['group_absent'] is True and terminal['cleanup']['group_absent'] is True
            and terminal['cleanup']['reaped'] is True and terminal['cleanup']['errors'] == []
            and terminal['error'] is None and terminal['clock_error'] is None, 'failed exit with completed cleanup, no timeout')
    require(terminal['command'] == [plan['python_executable'], '-u', str(ROOT/'scripts/collect_otto_coverage.py'),
            '--plan', str(BASE/'collection-plan-01.json'), '--plan-sha256', PINS['collection-plan-01.json'],
            '--output', str(BASE/'collection-01'), '--supervision', str(BASE/'collection-process-01.launch.json')]
            and terminal['cwd'] == str(ROOT), 'parent literal command and collection identity')
    require(terminal['pid'] == terminal['pgid'] != terminal['parent_pid']
            and terminal['clock_source_sha256'] == plan['sources']['src/openjev/research/suspend_clock.py']
            and terminal['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']
            and terminal['cap_seconds'] == plan['limits']['native_seconds'] == 900
            and terminal['deadline_ns'] == terminal['started_ns']+900*10**9
            and terminal['elapsed_ns'] == terminal['finished_ns']-terminal['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns']/1e9
            and terminal['started_ns'] < terminal['finished_ns'] < terminal['deadline_ns'], 'native parent timing and source witnesses')
    require(worker['completed_episodes'] == 72 and worker['pending'] == []
            and all(worker[key] == 0 for key in ('external_model_calls', 'optimizer_updates', 'target_readouts')),
            'all collection episodes complete, no pending operation or learning')
    seeds, first = (10101, 10102, 10103), {'lambda3': 13100001, 'lambda4': 13200001}
    expected = []
    for ri, (regime, start) in enumerate(first.items()):
        for case in range(12):
            offset = (ri*12+case) % 3
            for collector in seeds[offset:]+seeds[:offset]:
                expected.append((regime, start+case, 1+case % 3, collector, case))
    rows = [json.loads(line) for line in (BASE/'collection-01/collection-episodes.jsonl').read_text().splitlines()]
    require(len(rows) == 72, 'all72 episode records')
    counts, details = collections.Counter(), collections.defaultdict(list)
    found = 0
    for row, identity in zip(rows, expected, strict=True):
        regime, seed, hit, collector, case = identity
        episode = f'train:{regime}:{seed}:mlp8@{collector}'
        require(tuple(row[k] for k in ('regime', 'seed', 'initial_hit', 'collector_seed', 'case')) == identity
                and row['episode_id'] == episode and row['stage'] == 'train_collection', 'exact rotating trajectory chronology')
        steps = row['steps']
        require(type(steps) is int and 1 <= steps <= 2188 and type(row['found']) is bool
                and (row['found'] or steps == 2188) and row['updates'] == steps
                and row['final_update_assimilated'] is True and row['training_labels_generated'] is False
                and row['blocked_steps'] == 0, 'all found/censored pre-action intervals, final update and no labels')
        packet = row['final_public']
        require(packet['step'] == steps and packet['done'] is row['found']
                and ((packet['hit'] == -2 and packet['valid_actions'] == []) if row['found']
                     else (packet['hit'] in (0, 1, 2, 3) and bool(packet['valid_actions']))), 'terminal versus censored final packet')
        cell = regime, hit, collector
        counts[cell] += steps
        details[cell].append({'episode_id': episode, 'steps': steps, 'found': row['found'],
                              'preaction_indices': [0, steps-1]})
        found += int(row['found'])
    total_steps = sum(counts.values())
    require(set(worker['calls']) == {'model_load', 'native_reset', 'native_step', 'value_forward'}, 'only declared worker call channels')
    for name, expected_count in {'model_load': 3, 'native_reset': 74, 'native_step': total_steps, 'value_forward': total_steps}.items():
        record = worker['calls'][name]
        require(record['attempted'] == record['returned'] == expected_count
                and math.isfinite(record['seconds']) and record['seconds'] >= 0, 'worker ledger count reconciles to all episode lengths')
    require(total_steps <= plan['limits']['native_steps'] == 157536, 'native step cap')
    quotas = plan['quota_table']
    cells = [(regime, hit) for regime in first for hit in (1, 2, 3)]
    require([(q['regime'], q['initial_hit']) for q in quotas] == cells
            and sum(q['original_rows'] for q in quotas) == 5589, 'six fixed original strata')
    allocated = {cell: 2790*q['original_rows']//5589 for cell, q in zip(cells, quotas, strict=True)}
    order = sorted(cells, key=lambda cell: (-(2790*quotas[cells.index(cell)]['original_rows'] % 5589), cell))
    for cell in order[:2790-sum(allocated.values())]:
        allocated[cell] += 1
    comparison = []
    for q in quotas:
        cell = q['regime'], q['initial_hit']
        require(q['student_rows'] == allocated[cell] and q['teacher_rows'] == q['original_rows']-allocated[cell], 'independent largest-remainder quota')
        for index, collector in enumerate(seeds):
            planned = allocated[cell]//3 + int(index < allocated[cell] % 3)
            require(q['collector_rows'][str(collector)] == planned and len(details[(*cell, collector)]) == 4, 'collector quota and fixed four episodes')
            available = counts[(*cell, collector)]
            comparison.append({'regime': cell[0], 'initial_hit': cell[1], 'collector_seed': collector,
                               'planned': planned, 'available': available, 'shortfall': max(0, planned-available),
                               'sufficient': available >= planned, 'episodes': details[(*cell, collector)]})
    deficient = [row for row in comparison if not row['sufficient']]
    require(deficient, 'fixed-quota failure independently reproduced from metadata')
    forbidden = ['receipt.json', 'summary.json', 'selection.json', 'mixture-data.npz', 'mixture-rows.jsonl']
    require(not any((BASE/'collection-01'/name).exists() for name in forbidden), 'no completed/admitted mixture publication')
    result = {'version': 'otto-coverage-failure-review-v1', 'agreement': True, 'scope': SCOPE,
              'collection_status': 'failed_fixed_quota', 'completed_episodes': 72, 'native_steps': total_steps,
              'found': found, 'censored': 72-found, 'planned_student_rows': 2790, 'planned_teacher_rows': 2799,
              'planned_mixture_rows': 5589, 'cells': comparison, 'deficient_cells': deficient,
              'sum_cell_shortfalls': sum(r['shortfall'] for r in comparison),
              'quota_cells_sufficient': len(comparison)-len(deficient), 'quota_cells_total': 18,
              'worker_ledger': worker['calls'], 'worker_pending': worker['pending'],
              'source_manifest_entries': len(plan['sources']), 'source_hash_scope': 'Recorded failed-worker source map equals frozen plan; no source files reread.',
              'process': {'original_tool_session': 68486, 'original_tool_terminal_chunk': '1233b9', 'original_tool_exit_code': 1,
                          'tool_witness_source': 'Parent agent supplied original tool completion identity.',
                          **{k: terminal[k] for k in ('pid','pgid','parent_pid','status','returncode','timed_out','group_absent',
                              'clock_backend','started_ns','finished_ns','deadline_ns','elapsed_ns','wall_seconds','cleanup')}},
              'mixture_published': False, 'completed_collection_audit_executed': False,
              'learning_or_evaluation_admitted': False, 'retry_performed': False,
              'inputs': inputs}
    write('summary.json', result)
    report = ['# Preserved collection quota failure', '',
              f'The fixed 72-trajectory collection finished {total_steps:,} steps, then failed its required row quotas. '
              f'{len(deficient)} of 18 stratum/collector cells were short. No mixture, training or evaluation was admitted.', '',
              '| Setting | Initial hit | Collector | Available | Required | Shortfall |',
              '|---|---:|---:|---:|---:|---:|']
    for row in comparison:
        report.append(f"| {row['regime']} | {row['initial_hit']} | {row['collector_seed']} | {row['available']} | {row['planned']} | {row['shortfall']} |")
    report += ['', f'All 72 episode identities follow the frozen rotation, with four episodes per cell. The count for each episode is its number of pre-action states, indices 0 through steps minus one. Found and censored final updates remain included in the recorded episode lifecycle. The metadata records {found} found and {72-found} censored trajectories; these are collection descriptors, not an efficacy comparison.', '',
               'The original parent ended with exit 1 after 131.561186667 seconds. It did not time out, reaped its child and recorded the process group absent. The worker reported no pending operations: 3 model loads, 74 resets, and 52,072 matched forward/step calls. The saved traceback identifies the fixed reservoir quota check during mixture preparation.', '',
               'The original completed-collection auditor was not run. No replacement, resampling, seed change or retry was made. This readback opens only the plan, failure receipt, episode metadata and parent terminal. It does not decode checkpoint/NPZ arrays or replay trajectories. Raw numerical and detailed operation-journal correctness remains unaudited for this failed collection.', '',
               'Input identities and all 72 per-cell episode memberships are recorded in summary.json. Original source and output files are unchanged.', '']
    with (OUT/'report.md').open('x') as stream:
        stream.write('\n'.join(report))
    require(inputs == {name: desc(BASE/name) for name in PINS}, 'original metadata unchanged after readback')
    elapsed = time.monotonic()-started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
    require(elapsed < 60 and rss < 512*1024**2, 'bounded metadata reader')
    write('receipt.json', {'version': result['version'], 'status': 'completed', 'agreement': True, 'scope': SCOPE,
                          'source': desc(pathlib.Path(__file__).resolve()), 'inputs': inputs,
                          'elapsed_seconds': elapsed, 'review_clock': 'monotonic wall interval, not original native run time',
                          'peak_rss_bytes': rss, 'limits': {'seconds': 60, 'rss_bytes': 512*1024**2},
                          'files': {name: desc(OUT/name) for name in ('summary.json', 'report.md')}})
    print(json.dumps({'deficient_cells': [{k: row[k] for k in ('regime','initial_hit','collector_seed','available','planned','shortfall')} for row in deficient],
                      'sufficient': 18-len(deficient), 'total': 18, 'native_steps': total_steps, 'receipt': desc(OUT/'receipt.json')}))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write('review-failed.json', {'status': 'failed', 'error': repr(error), 'scope': SCOPE})
        raise
