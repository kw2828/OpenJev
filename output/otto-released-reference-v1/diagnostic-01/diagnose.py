"""Outcome-selected saved-only description of reference case 860074; no policy calls."""
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import sys
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = OUT.parent
PINS = {'plan-01.json': '85b64b07a37f40766f8e890170ed102035af01f4cd3611dfcc80cce6d7cfe1f4',
        'run-01/receipt.json': 'd360c63525d72a289c7ade4ec09377769ee4156bad2402e6185a1808d337d154',
        'audit-01/receipt.json': '0ce85a845994a3ce9877876e1f2ebf6774131ce390580948412051708b2515aa'}
CLOCK = ROOT / 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
TERMINAL_PIN = '306ad30cb07748e30c6292b7f3cf85cc83db3d4d21e3715f3ff7179469b0abb1'
ARMS = ('released_tf', 'analytic_all4', 'analytic_inbounds')
SCOPE = ('Outcome-selected descriptive diagnosis of the single censored case860074 and its two paired controls. '
         'Independent public posterior reconstruction from the frozen kernel, without actor/model/policy/simulator calls. '
         'Recorded costs and model values remain producer evidence. No episode removal, altered gates, action intervention, '
         'new trajectory, efficacy claim or model-value recomputation. Conditional entropy normalizes only a diagnostic copy.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def descriptor(path):
    return {'sha256': sha(path), 'bytes': path.stat().st_size}


def main():
    require({p.name for p in OUT.iterdir()} == {'diagnose.py'}, 'exclusive fresh diagnostic directory')
    receipt = {'status': 'started', 'scope': SCOPE, 'model_calls': 0, 'policy_calls': 0, 'simulator_calls': 0,
               'limits': {'seconds': 60, 'rss_bytes': 1024**3, 'output_bytes': 16*1024**2},
               'source': descriptor(Path(__file__)), 'inputs': PINS}
    clock = start = None
    def timeout(*_):
        raise TimeoutError('saved diagnostic 60-second cap')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        require(sha(CLOCK) == CLOCK_PIN, 'clock bytes')
        spec = importlib.util.spec_from_file_location('_diagnostic_native_clock', CLOCK)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        clock = mod.SuspendClock()
        start = clock.now_ns()
        def check():
            require(clock.now_ns() - start < 60*10**9, 'native diagnostic deadline')
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
            require(rss <= 1024**3, 'diagnostic RSS')
            require(sum(p.stat().st_size for p in OUT.iterdir()) <= 16*1024**2, 'diagnostic storage')
            receipt['peak_rss_bytes'] = rss
        write(OUT / 'started.json', receipt)
        for name, pin in PINS.items():
            require(sha(BASE / name) == pin, 'external input pin')
        plan, done, audit = (read(BASE / name) for name in PINS)
        require(done['status'] == audit['status'] == 'completed' and audit['agreement'] is True
                and audit['worker_sha256'] == PINS['run-01/receipt.json']
                and audit['plan_sha256'] == done['plan_sha256'] == PINS['plan-01.json'], 'complete audited run joins')
        for name, pin in plan['sources'].items():
            require(sha(ROOT / name) == pin, 'unchanged frozen sources')
        for rel, record in (('run-01', done), ('audit-01', audit)):
            folder = BASE / rel
            require({p.name for p in folder.iterdir()} == set(record['files']) | {'receipt.json'}, 'closed undemoted evidence')
            for name, desc in record['files'].items():
                require(descriptor(folder / name) == desc, 'payload identity before decoding')
                check()
        started = read(BASE / 'run-01/started.json')
        launch = Path(started['request']['supervision'])
        terminal_path = launch.with_name(launch.name.replace('.launch.json', '.terminal.json'))
        require(sha(launch) == done['supervision_sha256'] and sha(terminal_path) == TERMINAL_PIN
                and audit['terminal_sha256'] == TERMINAL_PIN, 'external process bindings')
        terminal = read(terminal_path)
        require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['group_absent'] is True
                and terminal['timed_out'] is False and terminal['cleanup']['errors'] == []
                and terminal['started_ns'] <= done['started_ns'] <= done['finished_ns'] <= terminal['finished_ns'] < terminal['deadline_ns'], 'successful native closure')
        qdesc = plan['inputs']['qualification_plan']
        require(descriptor(ROOT / qdesc['path']) == {k: qdesc[k] for k in ('sha256', 'bytes')}, 'qualified model/kernel plan')
        kernel_desc = read(ROOT / qdesc['path'])['inputs']['shift_kernel']
        kernel_path = ROOT / kernel_desc['path']
        require(descriptor(kernel_path) == {k: kernel_desc[k] for k in ('sha256', 'bytes')}, 'public shift kernel')
        receipt['kernel'] = kernel_desc
        receipt['terminal_sha256'] = TERMINAL_PIN
        os.environ.update({k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})
        import numpy as np

        with np.load(kernel_path, allow_pickle=False) as f:
            kernel = f['likelihood']
        require(kernel.dtype == np.float64 and kernel.shape == (4,107,107), 'kernel geometry')
        def selected(name):
            with (BASE / 'run-01' / name).open() as stream:
                for line in stream:
                    row = json.loads(line)
                    context = row.get('context', row)
                    if context.get('cohort') == 'shift' and context.get('seed') == 860074:
                        yield row
        forwards = {r['context']['step']: r for r in selected('forwards.jsonl')}
        episodes = {r['arm']: r for r in selected('episodes.jsonl')}
        require(set(episodes) == set(ARMS) and len(forwards) == 2188, 'all paired episode/forward membership')
        def update(p, public):
            x,y = public['position']
            q = p.copy()
            if public['done']:
                q.fill(0); q[x,y] = 1.
            else:
                q[x,y] = 0.
                q *= kernel[public['hit'],53-x:106-x,53-y:106-y]
                q[(q < 0) & (q > -1e-15)] = 0
                mass = q.sum()
                if mass > 1e-10:
                    q /= mass
            return q
        def stats(p):
            mass = float(p.sum()); positive = p[p > 0]
            entropy = None if mass == 0 else float(-np.sum((positive / mass) * np.log2(positive / mass)))
            return {'mass': mass, 'positive_support': int(len(positive)), 'maximum_cell_mass': float(p.max()),
                    'conditional_entropy_bits': entropy, 'positive_minimum': float(positive.min()) if len(positive) else None}
        states, publics, records = {}, {}, {a: [] for a in ARMS}
        with (OUT / 'timeline.jsonl').open('x') as output:
            for row in selected('transitions.jsonl'):
                check()
                arm, public = row['arm'], row['public']
                if row['kind'] == 'reset':
                    p = np.ones((53,53), np.float64) / 2808
                    states[arm] = update(p, public); publics[arm] = public
                    expected = row['posterior_after']
                    require(sha_bytes(states[arm]) == expected['sha256'], 'reset posterior identity')
                    continue
                p, before = states[arm], publics[arm]
                require(sha_bytes(p) == row['posterior_before']['sha256'], 'causal saved posterior')
                q = update(p, public)
                require(sha_bytes(q) == row['posterior_after']['sha256'], 'independent completed posterior identity')
                dtype = np.float32 if arm == 'released_tf' else np.float64
                costs = np.asarray([math.inf if v is None else v for v in row['costs']], dtype=dtype)
                ties = np.flatnonzero(np.abs(costs - costs.min()) < 1e-10).tolist()
                require(ties[0] == row['action'], 'unchanged exact choice')
                legal = before['valid_actions']
                record = {'arm': arm, 'step': row['step'], 'position_before': before['position'], 'position_after': public['position'],
                          'hit': public['hit'], 'done': public['done'], 'action': row['action'], 'costs': row['costs'],
                          'ties': ties, 'blocked': row['blocked'], 'stuck': row['stuck'], 'inbounds_actions': legal,
                          'best_inbounds_minus_selected_cost': float(min(float(costs[a]) for a in legal) - float(costs[row['action']])),
                          'before': stats(p), 'after': stats(q)}
                if arm == 'released_tf':
                    f = forwards[row['step']]
                    masses, values = np.asarray(f['branch_masses'], np.float32), np.asarray(f['values'], np.float32).reshape(4,4)
                    chosen = row['action']
                    record.update(branch_mass_min=float(masses.min()), branch_mass_max=float(masses.max()),
                                  branch_mass_floors=int(np.count_nonzero(masses == np.float32(1e-10))),
                                  selected_branch_masses=masses[chosen].tolist(), selected_branch_values=values[chosen].tolist(),
                                  all_value_min=float(values.min()), all_value_max=float(values.max()))
                output.write(json.dumps(record, allow_nan=False, sort_keys=True)+'\n')
                records[arm].append(record)
                states[arm], publics[arm] = q, public
        result = {'scope': SCOPE, 'case': {'cohort': 'shift', 'seed': 860074, 'block': 6, 'initial_hit': 1}, 'arms': {}}
        for arm, items in records.items():
            require(len(items) == episodes[arm]['steps'], 'complete paired trace')
            boundary = [r for r in items if len(r['inbounds_actions']) < 4]
            blocked = [r for r in items if r['blocked']]
            zero = [r for r in items if r['after']['mass'] == 0]
            tiny = [r for r in items if r['after']['mass'] <= 1e-10]
            runs = []
            for r in items:
                key = (r['action'], r['blocked'], tuple(r['position_after']))
                if runs and runs[-1]['key'] == key:
                    runs[-1]['last'] = r['step']; runs[-1]['length'] += 1
                else:
                    runs.append({'key': key, 'first': r['step'], 'last': r['step'], 'length': 1})
            anchors = {1,len(items)}
            for choices in (boundary,blocked,zero,tiny):
                if choices:
                    t = choices[0]['step']; anchors.update(range(max(1,t-2),min(len(items),t+2)+1))
            result['arms'][arm] = {'steps': len(items), 'found': items[-1]['done'], 'first_boundary_decision': boundary[0]['step'] if boundary else None,
                'first_blocked': blocked[0]['step'] if blocked else None, 'blocked_count': len(blocked),
                'first_zero_posterior': zero[0]['step'] if zero else None, 'first_mass_at_most_epsilon': tiny[0]['step'] if tiny else None,
                'minimum_mass': min(r['after']['mass'] for r in items), 'minimum_positive_support': min(r['after']['positive_support'] for r in items),
                'action_counts': dict(Counter(r['action'] for r in items)), 'tie_size_counts': dict(Counter(len(r['ties']) for r in items)),
                'longest_identical_action_position_run': max(runs,key=lambda r:r['length']),
                'selected_anchors': [items[t-1] for t in sorted(anchors)]}
            if blocked:
                result['arms'][arm]['blocked_legal_cost_gap_range'] = [min(r['best_inbounds_minus_selected_cost'] for r in blocked), max(r['best_inbounds_minus_selected_cost'] for r in blocked)]
                result['arms'][arm]['blocked_zero_mass_count'] = sum(r['before']['mass']==0 for r in blocked)
        audit_summary = read(BASE / 'audit-01/summary.json')
        means = audit_summary['cohorts']['shift']['means']
        w = audit_summary['cohorts']['shift']['initial_hit_weights']['1']/32
        gap = means['released_tf']['capped_time']-means['analytic_all4']['capped_time']
        contribution = w*(episodes['released_tf']['steps']-episodes['analytic_all4']['steps'])
        result['unchanged_weighted_decomposition'] = {'full_shift_gap':gap,'selected_case_contribution':contribution,'all_other95_contribution_without_reweighting':gap-contribution}
        result['unchanged_overall_flags'] = {k:audit_summary[k] for k in ('competent_reference','stronger_value_teacher','utility_compute_advantage')}
        write(OUT / 'summary.json', result)
        for name,pin in PINS.items():
            require(sha(BASE/name)==pin,'unchanged external evidence')
        check(); finish=clock.now_ns()
        receipt.update(status='completed',clock_backend=clock.backend,started_ns=start,finished_ns=finish,wall_seconds=(finish-start)/1e9,
                       cases=1,paired_arms=3,reconstructed_updates=sum(len(x) for x in records.values()),
                       files={p.name:descriptor(p) for p in OUT.iterdir()})
        write(OUT/'receipt.json',receipt);check()
        print(json.dumps({'status':'completed','receipt_sha256':sha(OUT/'receipt.json'),'wall_seconds':receipt['wall_seconds']}))
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0)
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        if (OUT/'receipt.json').exists():
            (OUT/'receipt.json').rename(OUT/'invalid-receipt.json')
        write(OUT/'failed.json',receipt)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


def sha_bytes(p):
    return hashlib.sha256(p.tobytes()).hexdigest()


if __name__=='__main__':
    main()
