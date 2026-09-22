"""Exploratory saved continuation-target noise; never a learning admission gate."""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import signal
import struct
import sys
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = 'scripts/diagnose_otto_target_noise.py'
BASE = 'output/otto-teacher-learning-v1/'
VERSION = 'otto-target-noise-v1'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
PINS = {
    'empirical_plan': (BASE+'plan-01.json', '7c155b10b630ee6fb7b16e302accfc0883b4a39ff7a0cb6f94fe4b6ffb0bec90'),
    'worker': (BASE+'run-01/receipt.json', 'f55f19acc8b8b22ca936b78a1023835556b1ecb80b9bb84bbaa1b3dc661a0bf3'),
    'worker_parent': (BASE+'supervision-01.terminal.json', '4dd9a826c8bfd0747e429ac0f3471b5e1614b03fe78f16f87e6a7834e19ebdf7'),
    'audit': (BASE+'audit-02/receipt.json', '3ef858eb393a2896a48e9ec7bd9a9feb4600a3c4c3111b8ec3484829a56826d0'),
    'audit_plan': (BASE+'audit-plan-02.json', '7ef4e5e195fb7e7c09c8ac3e02ffad1f8cfb37dd96d297712fba9143c4c201d5'),
    'audit_parent': (BASE+'audit-supervision-02.terminal.json', 'b0278cdd9d97f20683c9026da8ba5083db71d1661b81f768a2cf8fc3254c0270'),
}
LIMITS = {'seconds': 120, 'rss_bytes': 1024**3, 'output_bytes': 64*1024**2}
THREADS = dict.fromkeys(('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'), '1')
METHOD = {
    'scope': 'Exploratory all-558 saved-label diagnosis; no performance claim, pass gate, confidence intervals or independent-anchor assumption.',
    'rows': 558, 'episodes': 144, 'replicates': list(range(16)), 'horizon': 2188,
    'centering': 'Per replicate subtract unweighted mean over geometrically eligible actions. Use exact integer numerators A*C-sum(C), divide only at final moments.',
    'observed': 'Unweighted eligible-action mean of squared full-16 centered cost means, in moves squared.',
    'noise': 'Eligible-action mean of centered replicate sample variance (ddof=1) divided by 16.',
    'signal': 'Observed moment minus estimated noise, retained without truncation even if negative.',
    'weights': 'Original TRAIN row weight 558/(144*n_episode); group means renormalize these same weights within all/lambda3/lambda4.',
    'split': 'IDs 0-7 versus 8-15; action-mean cross moment and correlation cross/sqrt(left_second*right_second); undefined when either second moment is zero.',
    'ties': 'Exact equality of integer half-panel action cost sums; report first-numeric-action agreement, tied-set equality and overlap, with all rows retained.',
    'assumptions': 'Variance-of-mean and cross-moment interpretations require iid replicate vectors conditional on each fixed anchor; common randomness across actions is allowed. Anchors may be correlated.',
    'projection': 'Noise at 64 independent replicates projected as noise16/4; not measured precision or expected policy improvement.',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    def bad(value):
        raise ValueError('nonfinite JSON constant '+value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def read(path):
    return decode(path.read_bytes())


def digest(path, check=lambda: None):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'regular nonsymlink input')
    h, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            check()
            h.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': h.hexdigest()}


def write(path, value):
    with path.open('xb') as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def runtime():
    return {'executable': sys.executable, 'version': platform.python_version(), 'threads': THREADS}


def prepare(path):
    """Metadata/hash-only freeze. Do not decode panel costs or numerical arrays."""
    inputs = {}
    for role, (name, sha) in PINS.items():
        value = digest(ROOT/name)
        require(value['sha256'] == sha, 'external input pin '+role)
        inputs[role] = {'path': name, **value}
    receipt = read(ROOT/PINS['worker'][0])
    for name in ('panels.jsonl', 'training-data.json', 'training-data.npz'):
        value = digest(ROOT/BASE/'run-01'/name)
        require(value == receipt['files'][name], 'worker payload descriptor '+name)
        inputs[name] = {'path': BASE+'run-01/'+name, **value}
    sources = {name: digest(ROOT/name) for name in (SELF, CLOCK)}
    require(sources[CLOCK]['sha256'] == CLOCK_PIN, 'clock pin')
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, {'version': VERSION, 'status': 'frozen_before_panel_decode', 'method': METHOD,
                 'limits': LIMITS, 'runtime': runtime(), 'sources': sources, 'inputs': inputs})
    print(json.dumps({'plan': str(path), **digest(path)}))


def panel_moments(records, actions):
    require(type(actions) is list and actions and actions == sorted(set(actions))
            and all(type(a) is int and 0 <= a < 4 for a in actions), 'eligible actions')
    table = {}
    for r in records:
        require(type(r) is dict and set(r) == {'replicate_id', 'first_action', 'steps', 'found'}, 'record schema')
        rep, action, steps, found = (r[k] for k in ('replicate_id', 'first_action', 'steps', 'found'))
        require(type(rep) is int and 0 <= rep < 16 and type(action) is int and action in actions, 'record identity')
        require(type(steps) is int and 1 <= steps <= 2188 and type(found) is bool
                and (found or steps == 2188), 'complete capped continuation')
        require((rep, action) not in table, 'duplicate continuation')
        table[rep, action] = r
    require(len(table) == 16*len(actions), 'complete prescribed panel')
    costs = [[table[r, a]['steps'] for a in actions] for r in range(16)]
    n, a = 16, len(actions)
    numerators = [[a*c-sum(row) for c in row] for row in costs]
    totals = [sum(row[j] for row in numerators) for j in range(a)]
    means = [v/(n*a) for v in totals]
    noise = [(n*sum(row[j]**2 for row in numerators)-totals[j]**2)/(a*a*n*n*(n-1)) for j in range(a)]
    left = [sum(row[j] for row in numerators[:8])/(8*a) for j in range(a)]
    right = [sum(row[j] for row in numerators[8:])/(8*a) for j in range(a)]
    observed = math.fsum(x*x for x in means)/a
    noise_moment = math.fsum(noise)/a
    halves = [[sum(row[j] for row in part) for j in range(a)] for part in (costs[:8], costs[8:])]
    ties = [[actions[j] for j, value in enumerate(part) if value == min(part)] for part in halves]
    return {'actions': actions, 'costs_by_replicate': costs,
            'found_by_replicate': [[table[r, action]['found'] for action in actions] for r in range(16)],
            'centered_mean': means, 'noise_variance_of_mean': noise,
            'observed_second_moment': observed, 'noise_moment': noise_moment,
            'untruncated_signal_moment': observed-noise_moment,
            'split_centered_means': [left, right],
            'split_cross_moment': math.fsum(x*y for x, y in zip(left, right, strict=True))/a,
            'split_left_second_moment': math.fsum(x*x for x in left)/a,
            'split_right_second_moment': math.fsum(x*x for x in right)/a,
            'split_argmin_sets': ties, 'split_first_action_agreement': ties[0][0] == ties[1][0],
            'split_argmin_set_agreement': ties[0] == ties[1],
            'split_argmin_overlap': bool(set(ties[0]) & set(ties[1])),
            'split_left_tied': len(ties[0]) > 1, 'split_right_tied': len(ties[1]) > 1}


def aggregate(rows):
    weight = math.fsum(r['weight'] for r in rows)
    keys = ('observed_second_moment', 'noise_moment', 'untruncated_signal_moment',
            'split_cross_moment', 'split_left_second_moment', 'split_right_second_moment')
    result = {k: math.fsum(r['weight']*r[k] for r in rows)/weight for k in keys}
    denom = math.sqrt(result['split_left_second_moment']*result['split_right_second_moment'])
    result.update(rows=len(rows), episodes=len({r['episode_id'] for r in rows}), weight_sum=weight,
                  split_correlation=result['split_cross_moment']/denom if denom else None,
                  noise_to_observed_ratio=result['noise_moment']/result['observed_second_moment']
                  if result['observed_second_moment'] else None,
                  projected_noise_moment_at_64=result['noise_moment']/4,
                  negative_signal_rows=sum(r['untruncated_signal_moment'] < 0 for r in rows))
    for key in ('split_first_action_agreement', 'split_argmin_set_agreement', 'split_argmin_overlap',
                'split_left_tied', 'split_right_tied'):
        result[key] = {'count': sum(r[key] for r in rows),
                       'weighted_fraction': math.fsum(r['weight']*r[key] for r in rows)/weight}
    return result


def saved_weights(path, expected):
    """Read only four declared NPY weight columns, without NumPy/model imports."""
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), 'unique archive members')
        for kind in ('analytic', 'continuation'):
            for bits, dtype, code in ((64, '<f8', 'd'), (32, '<f4', 'f')):
                name = f'{kind}_weights_float{bits}.npy'
                info = archive.getinfo(name)
                require(info.file_size < 8192, 'bounded weight column')
                raw = archive.read(name)
                require(raw[:8] == b'\x93NUMPY\x01\x00', 'NPY v1 weight header')
                length = struct.unpack('<H', raw[8:10])[0]
                header = ast.literal_eval(raw[10:10+length].decode('latin1'))
                require(header == {'descr': dtype, 'fortran_order': False, 'shape': (558,)}, 'weight dtype/shape')
                require(raw[10+length:] == struct.pack('<'+code*558, *expected), 'original saved weights exact')


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    clock = None
    start = None
    receipt = {'version': VERSION, 'status': 'started', 'scientific_calls': 0, 'limits': LIMITS}
    def timeout(_signum, _frame):
        raise TimeoutError('120-second diagnostic deadline')
    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGTERM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 120)
    def check():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        require(rss <= LIMITS['rss_bytes'], 'RSS budget')
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'output budget')
        if clock is not None:
            require(clock.now_ns()-start <= 120*10**9, 'native elapsed budget')
    try:
        require(all(os.environ.get(k) == v for k, v in THREADS.items()), 'single-thread environment')
        require(digest(ROOT/CLOCK)['sha256'] == CLOCK_PIN, 'native clock pin')
        spec = importlib.util.spec_from_file_location('_target_noise_clock', ROOT/CLOCK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        clock = module.SuspendClock()
        start = clock.now_ns()
        receipt.update(started_ns=start, clock_backend=clock.backend)
        write(out/'started.json', receipt)
        require(digest(args.plan, check)['sha256'] == args.plan_sha256, 'external diagnostic plan pin')
        plan = read(args.plan)
        require(plan['version'] == VERSION and plan['method'] == METHOD and plan['limits'] == LIMITS
                and plan['runtime'] == runtime(), 'frozen method/runtime')
        for name, value in plan['sources'].items():
            require(digest(ROOT/name, check) == value, 'source identity')
        require(set(plan['sources']) == {SELF, CLOCK}, 'exact source closure')
        inputs = plan['inputs']
        require(set(inputs) == set(PINS) | {'panels.jsonl', 'training-data.json', 'training-data.npz'}, 'input roles')
        for role, value in inputs.items():
            require(digest(ROOT/value['path'], check) == {k: value[k] for k in ('sha256', 'bytes')}, 'input identity '+role)
            if role in PINS:
                require((value['path'], value['sha256']) == PINS[role], 'externally prescribed lineage')
        get = lambda role: read(ROOT/inputs[role]['path'])
        empirical, worker, audit = get('empirical_plan'), get('worker'), get('audit')
        require(worker['status'] == audit['status'] == 'completed' and audit['agreement'] is True, 'completed independent agreement')
        require(worker['plan_sha256'] == inputs['empirical_plan']['sha256']
                and audit['audit_plan_sha256'] == inputs['audit_plan']['sha256'], 'plan joins')
        for role, source_role in (('plan', 'empirical_plan'), ('worker', 'worker'), ('terminal', 'worker_parent')):
            require(audit['producer_inputs'][role]['sha256'] == inputs[source_role]['sha256'], 'audit producer join')
        for role, output in (('worker_parent', 'run-01'), ('audit_parent', 'audit-02')):
            parent = get(role)
            require(parent['status'] == 'completed' and parent['returncode'] == 0 and parent['timed_out'] is False
                    and parent['cleanup']['group_absent'] is True, 'successful original parent')
            require(str(ROOT/BASE/output) in parent['command'], 'original output command')
        for name, sha in empirical['sources'].items():
            require(digest(ROOT/name, check)['sha256'] == sha, 'inherited source identity')
        audit_dir = ROOT/BASE/'audit-02'
        require({p.name for p in audit_dir.iterdir()} == set(audit['files']) | {'receipt.json'}, 'audit closed inventory')
        for name, value in audit['files'].items():
            require(digest(audit_dir/name, check) == value, 'completed audit payload')
        for name in ('panels.jsonl', 'training-data.json', 'training-data.npz'):
            require(worker['files'][name] == {k: inputs[name][k] for k in ('sha256', 'bytes')}, 'worker payload joins')
        # Numerical cost/weight decoding starts only after every preceding pin/join.
        metadata = get('training-data.json')
        rows = metadata['rows']
        require(rows == empirical['selections'] and len(rows) == 558 and metadata['episodes'] == 144, 'all frozen anchors')
        counts = collections.Counter(r['episode_id'] for r in rows)
        require(len(counts) == 144 and all(r['stage'] == 'dagger' and r['regime'] in ('lambda3', 'lambda4') for r in rows), 'TRAIN-only cohort')
        weights = [558/(144*counts[r['episode_id']]) for r in rows]
        saved_weights(ROOT/inputs['training-data.npz']['path'], weights)
        result = []
        with (ROOT/inputs['panels.jsonl']['path']).open('rb') as stream:
            for i, row in enumerate(rows):
                check()
                require(row['anchor_id'] == i, 'canonical anchor order')
                attempt, returned = decode(next(stream)), decode(next(stream))
                require(attempt == {'event': 'attempt', 'phase': 'sampling', 'anchor_id': i}, 'panel attempt')
                require(returned['event'] == 'return' and returned['phase'] == 'sampling' and returned['anchor_id'] == i, 'panel return')
                value = panel_moments(returned['records'], row['public']['valid_actions'])
                value.update(anchor_id=i, episode_id=row['episode_id'], regime=row['regime'], weight=weights[i],
                             source_row_index=row['row_index'], prefix_index=row['prefix_index'])
                result.append(value)
            require(stream.read() == b'', 'all 558 panels exactly')
        require(sum(16*len(r['actions']) for r in result) == 32304, 'all continuation records')
        with (out/'rows.jsonl').open('xb') as stream:
            for row in result:
                check()
                stream.write(encoded(row))
            stream.flush()
            os.fsync(stream.fileno())
        summary = {'version': VERSION, 'method': METHOD,
                   'groups': {g: aggregate([r for r in result if g == 'all' or r['regime'] == g]) for g in ('all', 'lambda3', 'lambda4')},
                   'rows': 558, 'continuations': 32304, 'scope': METHOD['scope']}
        write(out/'summary.json', summary)
        # Close identities again without regenerating models, targets or rollouts.
        for role, value in inputs.items():
            require(digest(ROOT/value['path'], check) == {k: value[k] for k in ('sha256', 'bytes')}, 'final input identity '+role)
        for name, value in plan['sources'].items():
            require(digest(ROOT/name, check) == value, 'final source identity')
        check()
        receipt.update(status='completed', plan_sha256=args.plan_sha256, inputs=inputs, sources=plan['sources'],
                       rows=558, continuations=32304, weight_columns_verified=4,
                       inherited_source_count=len(empirical['sources']), limitations=[
                           'Prior completed audit authenticated, not rerun; panel event streams and neural scores are not regenerated.',
                           METHOD['assumptions'], METHOD['scope']])
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - preserve the original failed attempt
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
    finally:
        try:
            receipt['files'] = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
            if clock is not None and start is not None:
                finished = clock.now_ns()
                receipt.update(finished_ns=finished, wall_seconds=(finished-start)/1e9)
            receipt['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
            if receipt['status'] == 'completed':
                check()
            write(out/'receipt.json', receipt)
            if receipt['status'] == 'completed':
                check()
        except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - demote late completion failures
            signal.setitimer(signal.ITIMER_REAL, 0)
            if (out/'receipt.json').exists():
                (out/'receipt.json').rename(out/'receipt.invalid.json')
            receipt.update(status='failed', publication_error=repr(error))
            write(out/'failed.json', {'error': repr(error), 'traceback': traceback.format_exc()})
            receipt['files'] = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
            write(out/'receipt.json', receipt)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({'status': receipt['status'], 'receipt': digest(out/'receipt.json')}))
    return 0 if receipt['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('plan', 'run'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'plan':
        prepare(args.plan.resolve())
        return 0
    require(args.output is not None and args.plan_sha256 is not None, 'run requires output and external plan pin')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
