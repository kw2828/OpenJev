"""Saved spatial-control qualification and autonomous audit, with counted readouts.

All actor values are replayed through the separately qualified NumPy head code.
That network algebra is shared with producer NumPy scoring, not a third network
implementation. Public filtering, sixteen observation branches, exact choices,
coverage, weighting, gates and cost accounting are independently reconstructed.
Torch execution, native randomness, historical data truth and timing remain
source-bound evidence. No optimization, native environment or new collection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = 'scripts/audit_otto_conditioning_control.py'
HELPER_PIN = '094fd41e7b9c34d0df580c1d99aef0e2f1bc2c83997afad36cd8c15f14794e1b'
MODEL = 'src/openjev/research/otto_spatial_value.py'
MODEL_PIN = '1b57a7e46edd81ad3d8459af4d90bd2b8170f19a121ab7ef8a2b768ea706f857'
if hashlib.sha256((ROOT/HELPER).read_bytes()).hexdigest() != HELPER_PIN:
    raise ValueError('pinned independent public-state audit helper')
_spec = importlib.util.spec_from_file_location('_spatial_control_independent', ROOT/HELPER)
C = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = C
_spec.loader.exec_module(C)
E, B, require = C.E, C.B, C.require
VERSION = 'otto-spatial-control-saved-audit-v1'
RUNNER = 'scripts/study_otto_spatial_control.py'
SEEDS = (10101, 10102, 10103)
FAMILIES = ('spatial', 'neighbor_free', 'cnn', 'dense128', 'statistics')
ARMS = tuple(f'{kind}@{seed}' for seed in SEEDS for kind in FAMILIES)+('analytic_inbounds',)
REGIMES = {'lambda3': 3., 'lambda4': 4., 'lambda5': 5.}
FIRST = {'lambda3': 16100001, 'lambda4': 16200001, 'lambda5': 16300001}
HORIZON, NTRAIN, NVALID = 2188, 5589, 1109
TIMES, METRICS = E.TIMES, E.METRICS
ATOL, RTOL = 1e-8, 1e-10
MAX_CALLS = {'qualify': 780, 'study': 2363040}
LIMITS = {
    'qualify': {'native_seconds': 600, 'rss_bytes': 4*1024**3, 'output_bytes': 256*1024**2},
    'study': {'native_seconds': 21600, 'rss_bytes': 4*1024**3, 'output_bytes': 1024**3},
}
THREADS = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
SCOPE = __doc__


def require_threads():
    require(all(os.environ.get(k) == v for k, v in THREADS.items()), 'CPU1 environment before model import')


def compact_bytes(record):
    return (json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n').encode()


def journal_projection(mode):
    """Serialize bounded maximum-width audit events, not empirical outcomes."""
    calls = MAX_CALLS[mode]
    # [call, event, arm index, episode-or-tuple index, step, rows] and
    # [call, return, elapsed nanoseconds]. All admitted values are bounded here.
    attempt = compact_bytes([calls, 0, 14, 1151, HORIZON, 16])
    returned = compact_bytes([calls, 1, LIMITS[mode]['native_seconds']*10**9])
    result = {'calls': calls, 'attempt_bytes': len(attempt), 'return_bytes': len(returned),
              'maximum_journal_bytes': calls*(len(attempt)+len(returned)),
              'metadata_reserve_bytes': 32*1024**2,
              'scope': 'Two bounded compact JSON arrays per actual saved-head call; no duplicated branch arrays'}
    require(result['maximum_journal_bytes']+result['metadata_reserve_bytes'] < LIMITS[mode]['output_bytes'],
            'prospective full-cap audit journal fits allocation')
    return result


def evaluation_order():
    for ri, (regime, first) in enumerate(FIRST.items()):
        for case in range(24):
            offset = (ri*24+case) % len(ARMS)
            for arm in ARMS[offset:]+ARMS[:offset]:
                yield regime, first+case, 1+case % 3, arm, case//3


def payload_names(mode):
    require(mode in LIMITS, 'declared audit mode')
    names = {'started.json', 'runtime.json', 'inference-setup.json', 'work-contexts.jsonl', 'work.jsonl', 'summary.json',
             'serialization-projection.json'}
    return names | ({'preparation.json', 'parity.jsonl', 'qualification-arrays.npz', 'qualification-tuples.jsonl'} if mode == 'qualify' else
                    {'native-setup.json', 'eval-transitions.jsonl', 'eval-episodes.jsonl', 'evaluation.jsonl'})


def aggregate(rows, mixtures):
    require([(r['regime'], r['seed'], r['initial_hit'], r['arm'], r['block']) for r in rows]
            == list(evaluation_order()), 'complete ordered 1152-episode trial')
    for row in rows:
        require(type(row['found']) is bool and type(row['steps']) is int and 1 <= row['steps'] <= HORIZON
                and (row['found'] or row['steps'] == HORIZON) and row['updates'] == row['steps']
                and row['blocked_steps'] == 0 and row['final_update_assimilated'] is True, 'complete episode outcome')
        for metric in METRICS:
            E.finite(float(row[metric]) if metric == 'found' else row[metric], 'finite recorded metric')
        require(abs(row['controller_seconds']-math.fsum(row[k] for k in TIMES)) <= 1e-9, 'complete controller cost')
    panels, competence, improvement, baseline_competence = {}, [], [], []

    def add(group, name, value, threshold, passed):
        group.append({'name': name, 'value': value, 'threshold': threshold, 'passes': bool(passed)})

    for regime in REGIMES:
        weights = {int(k): v for k, v in mixtures[regime].items()}
        require(set(weights) == {1, 2, 3} and all(type(w) in (int, float) and math.isfinite(w) and 0 < w < 1 for w in weights.values())
                and abs(math.fsum(weights.values())-1) <= 1e-12, 'positive normalized mixture')
        selected = [r for r in rows if r['regime'] == regime]

        def mean(subset, metric, weights=weights):
            return math.fsum(weights[h]*math.fsum(float(r[metric]) for r in subset if r['initial_hit'] == h)
                             / sum(r['initial_hit'] == h for r in subset) for h in (1, 2, 3))

        means = {a: {m: mean([r for r in selected if r['arm'] == a], m) for m in METRICS} for a in ARMS}
        blocks = [{a: mean([r for r in selected if r['arm'] == a and r['block'] == b], 'steps') for a in ARMS} for b in range(8)]
        families = {f: {m: math.fsum(means[f'{f}@{s}'][m] for s in SEEDS)/3 for m in METRICS} for f in FAMILIES}
        for family, group in tuple((f, competence if f == 'spatial' else baseline_competence) for f in FAMILIES):
            for seed in SEEDS:
                fit = means[f'{family}@{seed}']
                add(group, f'{regime}.{family}.{seed}.success', fit['found'], .95, fit['found'] >= .95)
                bound = 1.05*means['analytic_inbounds']['steps']
                add(group, f'{regime}.{family}.{seed}.moves', fit['steps'], bound, fit['steps'] <= bound)
        candidate = families['spatial']
        for family in FAMILIES[1:]:
            control = families[family]
            wins = sum(math.fsum(b[f'{family}@{s}']-b[f'spatial@{s}'] for s in SEEDS)/3 > 0 for b in blocks)
            prefix = regime+'.'+family
            add(improvement, prefix+'.success', candidate['found'], control['found'], candidate['found'] >= control['found'])
            add(improvement, prefix+'.moves', candidate['steps'], .95*control['steps'], candidate['steps'] <= .95*control['steps'])
            add(improvement, prefix+'.positive_blocks', wins, 6, wins >= 6)
            add(improvement, prefix+'.cost', candidate['controller_seconds'], control['controller_seconds'],
                candidate['controller_seconds'] <= control['controller_seconds'])
        panels[regime] = {'weights': {str(k): v for k, v in weights.items()}, 'means': means,
                         'family_means': families, 'blocks': blocks,
                         'strata': {str(h): {a: {m: math.fsum(float(r[m]) for r in selected if r['arm'] == a and r['initial_hit'] == h)/8
                                               for m in METRICS} for a in ARMS} for h in (1, 2, 3)},
                         'raw_counts': {a: {'found': sum(r['found'] for r in selected if r['arm'] == a), 'episodes': 24} for a in ARMS}}
    return {'version': 'otto-spatial-control-v1', 'episodes': 1152, 'paired_cases': 72, 'regimes': panels, 'competence_checks': competence,
            'improvement_checks': improvement, 'control_competence_checks': baseline_competence,
            'pilot_continuation': all(r['passes'] for r in competence+improvement),
            'learned_architecture_advantage_established': False}


def amortization(result, setup, fit_seconds, preparation_seconds, module_seconds):
    E.finite(preparation_seconds, 'preparation cost')
    E.finite(module_seconds, 'module cost')
    scenarios = {}
    for regime, panel in result['regimes'].items():
        scenarios[regime] = {}
        for arm, metrics in panel['means'].items():
            if arm == 'analytic_inbounds':
                learn = preparation = deployment = 0.
                utility = metrics['controller_seconds']
            else:
                learn, preparation = fit_seconds[arm], preparation_seconds/15
                deployment = setup[arm]['seconds']+module_seconds/15
                utility = metrics['controller_seconds']-metrics['setup_allocation_seconds']
            for value in (learn, preparation, deployment, utility):
                E.finite(value, 'nonnegative amortization component')
            scenarios[regime][arm] = {'fit_seconds': learn, 'preparation_share_seconds': preparation,
                'deployment_setup_seconds': deployment, 'controller_without_deployment_seconds': utility,
                'seconds_per_search': {str(h): utility+(learn+preparation+deployment)/h for h in (1, 100, 10000)}}
    return scenarios


def branches(probability, position, kernel, np):
    """Independent action-major/hit-minor geometry and unchanged mass floor."""
    require(probability.shape == (53, 53) and probability.dtype == np.float64
            and np.isfinite(probability).all() and (probability >= 0).all(), 'public nonnegative float64 belief')
    require(len(position) == 2 and all(type(v) is int and 0 <= v < 53 for v in position), 'public integer position')
    require(kernel.shape == (4, 107, 107) and kernel.dtype == np.float64
            and np.isfinite(kernel).all() and (kernel >= 0).all() and (kernel <= 1).all()
            and not kernel[:, 53, 53].any(), 'same finite kernel and excluded found source')
    unnormalized = np.zeros((16, 105, 105), np.float64)
    normalized = np.zeros_like(unnormalized)
    raw = np.empty((4, 4), np.float64)
    successors = np.empty((16, 2), np.int64)
    for action in range(4):
        x, y = B.move(position, action)
        for hit in range(4):
            i = 4*action+hit
            joint = probability*kernel[hit, 53-x:106-x, 53-y:106-y]
            mass = float(joint.sum(dtype=np.float64))
            raw[action, hit] = mass
            successors[i] = (x, y)
            unnormalized[i, 52-x:105-x, 52-y:105-y] = joint
            normalized[i, 52-x:105-x, 52-y:105-y] = joint/max(mass, 1e-10)
    return unnormalized, normalized, successors, raw, np.maximum(raw, 1e-10)


def qualification_ids():
    """No state or outcome selection beyond the prospectively fixed tuples."""
    rows = [{'tuple_id': f'{split}:{i}', 'source': 'cache', 'split': split, 'row_index': i}
            for split in ('train', 'valid') for i in range(8)]
    for regime in REGIMES:
        for position in ('center', 'lower', 'upper'):
            for kind in ('asymmetric', 'point_successor', 'zero', 'subfloor'):
                rows.append({'tuple_id': f'{regime}:{position}:{kind}', 'source': 'synthetic', 'regime': regime,
                             'position': {'center': [26, 26], 'lower': [0, 0], 'upper': [52, 52]}[position],
                             'belief_kind': kind})
    return rows


def synthetic_belief(position_name, kind, np):
    q = {'center': [26, 26], 'lower': [0, 0], 'upper': [52, 52]}[position_name]
    x, y = np.indices((53, 53), dtype=np.int64)
    raw = (1+(17*x+29*y+7*x*y) % 97).astype(np.float64)
    raw[tuple(q)] = 0.
    probability = raw/raw.sum(dtype=np.float64)
    if kind == 'point_successor':
        probability.fill(0.)
        action = next(a for a in range(4) if B.move(q, a) != q)
        probability[tuple(B.move(q, action))] = 1.
    elif kind == 'zero':
        probability.fill(0.)
    elif kind == 'subfloor':
        probability *= np.float64(1e-12)
    else:
        require(kind == 'asymmetric', 'fixed synthetic belief family')
    return probability, q


def close_arrays(actual, expected, np, label, *, atol=ATOL, rtol=RTOL):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype == np.float64
            and np.isfinite(actual).all() and np.isfinite(expected).all(), label+' shape/type/finite')
    require(bool(np.all(np.abs(actual-expected) <= atol+rtol*np.abs(expected))), label+' tolerance')
    return float(np.max(np.abs(actual-expected))) if actual.size else 0.


def selected_public_cache(archive, count, np):
    """Decode only public-state columns, retaining the fixed first eight rows.

    A compressed NPZ column may decompress in full. Targets and legacy features
    are never requested, even for shape or numeric validation in qualification.
    """
    require(set(archive.files) == {'features', 'target', 'beliefs', 'positions', 'sensing_length'}, 'historical cache membership')
    result = {}
    for key, shape, dtype in (('beliefs', (count, 53, 53), np.float64), ('positions', (count, 2), np.int64),
                             ('sensing_length', (count,), np.float64)):
        value = archive[key]
        require(value.shape == shape and value.dtype == dtype, 'qualified public column shape/type')
        result[key] = value[:8].copy()
    return result


class Audit(C.Audit):
    """Reuse fixed public filtering/journal primitives, with new saved heads."""

    def __init__(self, args):
        super().__init__(args)
        require(args.mode in LIMITS, 'fixed audit mode')
        self.limits = LIMITS[args.mode]
        self.receipt.update(version=VERSION, limits=self.limits, scope=SCOPE,
                            readout_attempts=0, readout_returns=0, saved_checkpoint_readout_calls=0,
                            saved_network_rows=0, readout_seconds=0., pending=None,
                            journal_projection=journal_projection(args.mode))
        self.readout_context = (0, 0)
        self.model = None

    def arrays_close(self, actual, expected, name, *, tolerance=ATOL):
        delta = close_arrays(actual, expected, self.np, name, atol=tolerance)
        self.maximum_prediction_error = max(self.maximum_prediction_error, delta)
        self.c.count += actual.size

    def readout_event(self, record):
        data = compact_bytes(record)
        with (self.out/'readouts.jsonl').open('ab') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    def predicted(self, centered, successors, sensing, arm):
        """One actual sixteen-row saved-model call, journaled before execution."""
        self.check()
        require(arm in ARMS[:-1], 'declared learned replay arm')
        require(centered.shape == (16, 105, 105) and successors.shape == (16, 2), 'one exact branch batch')
        call = self.receipt['readout_attempts']+1
        require(call <= MAX_CALLS[self.args.mode], 'prospective saved-readout cap')
        context, step = self.readout_context
        require(type(context) is int and 0 <= context <= 1151 and type(step) is int and 0 <= step <= HORIZON,
                'bounded causal readout context')
        event = [call, 0, ARMS.index(arm), context, step, 16]
        self.receipt['readout_attempts'] = call
        self.receipt['pending'] = event
        self.readout_event(event)
        begin = self.clock.now_ns()
        value = self.heads[arm].normalized(centered, successors, self.np.full(16, sensing, self.np.float64))
        elapsed = self.clock.now_ns()-begin
        require(type(elapsed) is int and 0 <= elapsed <= self.limits['native_seconds']*10**9, 'bounded monotonic readout duration')
        self.receipt['readout_returns'] += 1
        self.receipt['saved_checkpoint_readout_calls'] += 1
        self.receipt['saved_network_rows'] += 16
        self.receipt['readout_seconds'] += elapsed/1e9
        self.readout_event([call, 1, elapsed])
        self.receipt['pending'] = None
        require(value.shape == (16,) and value.dtype == self.np.float64 and self.np.isfinite(value).all(),
                'finite signed saved-model normalized values')
        self.check()
        return value

    def numerical_inputs(self, plan):
        require_threads()
        import numpy as np
        self.np, self.maximum_prediction_error = np, 0.
        self.kernels, self.mixtures, self.heads, self.datasets = {}, {}, {}, {}
        self.spatial_run = self.path(plan, 'spatial_receipt').parent
        self.training_reference = B.read(self.spatial_run/'summary.json')
        self.baseline = B.read(self.spatial_run/'preparation.json')['c0_float32']
        require(type(self.baseline) is float and math.isfinite(self.baseline)
                and float(np.float32(self.baseline)) == self.baseline, 'authenticated float32 TRAIN baseline')
        self.c.equal(B.digest(ROOT/MODEL, self.check)['sha256'], MODEL_PIN, 'qualified saved-head source before import')
        self.model = B.load(ROOT/MODEL, '_spatial_control_audit_saved_heads')
        for regime in REGIMES:
            archive = self.load_arrays(self.path(plan, 'kernel_'+regime))
            self.c.equal(set(archive), {'likelihood', 'initial_hit_weights'}, 'exact kernel payload')
            kernel, weight = archive['likelihood'], archive['initial_hit_weights']
            require(kernel.shape == (4, 107, 107) and kernel.dtype == np.float64 and np.isfinite(kernel).all()
                    and (kernel >= 0).all() and (kernel <= 1).all() and not kernel[:, 53, 53].any(), 'unchanged frozen kernel')
            require(weight.shape == (4,) and weight.dtype == np.float64 and weight[0] == 0
                    and np.isfinite(weight).all() and (weight[1:] > 0).all()
                    and abs(float(weight.sum())-1) <= 1e-12, 'positive normalized initial mixture')
            self.kernels[regime], self.mixtures[regime] = kernel, {h: float(weight[h]) for h in (1, 2, 3)}
        for arm in ARMS[:-1]:
            archive = self.load_arrays(self.path(plan, 'head_'+arm.replace('@', '_')))
            self.c.equal(self.model.validate_head(archive), arm.split('@')[0], 'correct family checkpoint')
            self.c.equal(float(archive['c0']), self.baseline, 'unchanged common training baseline')
            self.heads[arm] = self.model.FrozenValue(archive)
        if self.args.mode == 'qualify':
            for split, count in (('train', NTRAIN), ('valid', NVALID)):
                with np.load(self.path(plan, split+'_data'), allow_pickle=False) as archive:
                    selected = selected_public_cache(archive, count, np)
                metadata = iter(B.lines(self.path(plan, split+'_rows'), self.check))
                selected['rows'] = [next(metadata) for _ in range(8)]
                for i, row in enumerate(selected['rows']):
                    require(row['row_index'] == i and row['stage'] == split, 'unchanged fixed cache prefix')
                    probability, position = selected['beliefs'][i], selected['positions'][i].tolist()
                    self.c.equal(position, row['public']['position'], 'cached public position')
                    require(np.isfinite(probability).all() and (probability >= 0).all(), 'finite public cached state')
                    self.c.equal(E.A.A.posterior_record(probability), row['posterior'], 'same historical posterior witness')
                self.datasets[split] = selected

    def inference_setup(self, plan):
        setup = B.read(self.args.run/'inference-setup.json')
        self.c.equal(set(setup), set(ARMS[:-1]), 'all fifteen independently loaded heads')
        for arm in ARMS[:-1]:
            row = setup[arm]
            path = self.path(plan, 'head_'+arm.replace('@', '_'))
            call = self.work.call('model_load', {'phase': 'inference_setup', 'arm': arm})
            self.c.equal((row['checkpoint'], row['sha256'], row['allocated_episodes']),
                         (str(path.relative_to(ROOT)), B.digest(path)['sha256'], 72 if self.args.mode == 'study' else 0),
                         'same final checkpoint and allocation denominator')
            for key in ('seconds', 'instrumented_seconds', 'excluded_io_seconds'):
                E.finite(row[key], 'finite measured setup duration')
            self.c.close(row['seconds'], row['instrumented_seconds']-row['excluded_io_seconds'], 'net complete setup')
            require(row['seconds']+1e-9 >= call['seconds'], 'outer setup includes counted load')
            self.c.equal(row['storage'], self.heads[arm].storage_bytes(), 'immutable deployed checkpoint storage')
        return setup

    def qualification_inputs(self):
        np, c = self.np, self.c
        arrays = self.load_arrays(self.args.run/'qualification-arrays.npz')
        shapes = {'beliefs': ((52, 53, 53), np.float64), 'positions': ((52, 2), np.int64),
                  'sensing_length': ((52,), np.float64), 'eligible_mask': ((52, 4), np.bool_),
                  'centered_u': ((52, 16, 105, 105), np.float64), 'centered_z': ((52, 16, 105, 105), np.float64),
                  'raw_masses': ((52, 4, 4), np.float64), 'weights': ((52, 4, 4), np.float64),
                  'successors': ((52, 16, 2), np.int64)}
        c.equal(set(arrays), set(shapes), 'all nine shared qualification arrays')
        for name, (shape, dtype) in shapes.items():
            require(arrays[name].shape == shape and arrays[name].dtype == dtype and np.isfinite(arrays[name]).all(),
                    'complete finite shared qualification '+name)
        require(np.array_equal(arrays['weights'], np.maximum(arrays['raw_masses'], 1e-10)), 'exact saved floor identity')
        prep = B.read(self.args.run/'preparation.json')
        seconds = E.finite(prep['seconds'], 'paid shared qualification preparation')
        descriptors = {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                          'sha256': hashlib.sha256(v.tobytes()).hexdigest(), 'bytes': v.nbytes}
                       for k, v in arrays.items()}
        c.equal(prep, {'seconds': seconds, 'tuples': 52,
                'array_file': B.digest(self.args.run/'qualification-arrays.npz', self.check), 'arrays': descriptors,
                'scope': 'Shared tuples once;16 fixed historical public prefixes plus36 prescribed synthetic tuples; targets not decoded.'},
                'exact shared preparation evidence')
        records = list(B.lines(self.args.run/'qualification-tuples.jsonl', self.check))
        c.equal(len(records), 52, 'every common qualification tuple')
        for index, identity in enumerate(qualification_ids()):
            self.check()
            record = dict(identity)
            if identity['source'] == 'cache':
                split, i = identity['split'], identity['row_index']
                data = self.datasets[split]
                metadata = data['rows'][i]
                probability, q, lam = data['beliefs'][i], data['positions'][i].tolist(), float(data['sensing_length'][i])
                regime = metadata['regime']
                c.equal(lam, REGIMES[regime], 'cached regime identity')
                require(metadata['public']['done'] is False, 'nonterminal historical decision prefix')
                record.update(regime=regime, episode_id=metadata['episode_id'], prefix_index=metadata['prefix_index'])
            else:
                regime = identity['regime']
                name = {(26, 26): 'center', (0, 0): 'lower', (52, 52): 'upper'}[tuple(identity['position'])]
                probability, q = synthetic_belief(name, identity['belief_kind'], np)
                lam = REGIMES[regime]
            allowed = [a for a in range(4) if B.move(q, a) != q]
            if identity['source'] == 'cache':
                c.equal(metadata['public']['valid_actions'], allowed, 'independent historical action eligibility')
            u, z, successors, raw, weights = branches(probability, q, self.kernels[regime], np)
            expected = {'beliefs': probability, 'positions': np.asarray(q, np.int64), 'sensing_length': np.float64(lam),
                        'eligible_mask': np.asarray([a in allowed for a in range(4)], np.bool_),
                        'centered_u': u, 'centered_z': z, 'successors': successors, 'raw_masses': raw, 'weights': weights}
            hashes = {}
            for name, value in expected.items():
                hashes[name] = hashlib.sha256(value.tobytes()).hexdigest()
                c.equal(hashlib.sha256(arrays[name][index].tobytes()).hexdigest(), hashes[name], 'exact independently rebuilt tuple '+name)
            record.update(index=index, position=q, sensing_length=lam, allowed_actions=allowed, array_sha256=hashes)
            c.equal(records[index], record, 'fixed tuple provenance/order and byte identities')
        return arrays, records, seconds

    def qualification(self, plan, worker, runtime):
        np, c = self.np, self.c
        require(worker['completed_episodes'] == 0 and worker['parity_passed'] is True, 'complete numerical-only qualification')
        setup = self.inference_setup(plan)
        arrays, records, seconds = self.qualification_inputs()
        stream = iter(B.lines(self.args.run/'parity.jsonl', self.check))
        maxima = {'values': 0., 'costs': 0.}
        for arm in ARMS[:-1]:
            self.work.call('parity_restore', {'phase': 'parity_restore', 'fit_id': arm})
            pin = plan['inputs']['head_'+arm.replace('@', '_')]['sha256']
            for index, record in enumerate(records):
                context = {'phase': 'parity', 'fit_id': arm, 'step': index}
                self.work.call('parity_numpy_forward', context)
                self.work.call('parity_torch_forward', context)
                self.readout_context = (index, index)
                values = 64*self.predicted(arrays['centered_z'][index], arrays['successors'][index],
                                           float(arrays['sensing_length'][index]), arm)
                costs = E.costs(values, arrays['weights'][index], np)
                row = next(stream)
                expected_fields = {'fit_id', 'tuple_id', 'tuple_index', 'checkpoint_sha256', 'array_sha256', 'allowed_actions',
                                   'values_numpy', 'values_torch', 'costs_numpy', 'costs_torch', 'action_numpy', 'action_torch',
                                   'maximum_error', 'passed'}
                c.equal(set(row), expected_fields, 'exact per-head parity schema')
                c.equal((row['fit_id'], row['tuple_id'], row['tuple_index'], row['checkpoint_sha256'],
                         row['array_sha256'], row['allowed_actions']),
                        (arm, record['tuple_id'], index, pin, record['array_sha256'], record['allowed_actions']),
                        'every fixed tuple/checkpoint/input join')
                saved = {}
                for suffix in ('numpy', 'torch'):
                    v, q = np.asarray(row['values_'+suffix], np.float64), np.asarray(row['costs_'+suffix], np.float64)
                    self.arrays_close(v, values, 'all replayed physical qualification values')
                    self.arrays_close(q, costs, 'all replayed four qualification costs')
                    self.arrays_close(q, E.costs(v, arrays['weights'][index], np), 'saved branch reduction')
                    action = row['action_'+suffix]
                    require(type(action) is int and action == E.choice(q.tolist(), record['allowed_actions'], True, np)
                            == E.choice(costs.tolist(), record['allowed_actions'], True, np), 'exact replayed eligible action')
                    saved[suffix] = (v, q)
                errors = {}
                for j, label in enumerate(('values', 'costs')):
                    errors[label] = close_arrays(saved['numpy'][j], saved['torch'][j], np, 'original physical NumPy/Torch predicate')
                    maxima[label] = max(maxima[label], errors[label])
                c.equal(row['maximum_error'], errors, 'exact retained numerical errors')
                require(row['passed'] is True, 'every original comparison passed')
        B.exhausted(stream, 'all780 ordered per-head comparisons')
        expected = {'model_load': 15, 'parity_restore': 15, 'parity_numpy_forward': 780, 'parity_torch_forward': 780}
        calls = self.finish_work(worker, expected)
        c.equal((self.receipt['saved_checkpoint_readout_calls'], self.receipt['saved_network_rows']), (780, 12480), 'all saved qualification readouts')
        result = {'version': 'otto-spatial-control-v1', 'mode': 'qualify', 'parity_records': 780, 'shared_tuples': 52,
                  'parity_passed': True, 'completed_episodes': 0, 'maximum_error': maxima, 'atol': ATOL, 'rtol': RTOL,
                  'learned_architecture_advantage_established': False}
        c.equal(B.read(self.args.run/'summary.json'), result, 'all original qualification outcomes')
        disjoint = (seconds+math.fsum(r['seconds'] for r in setup.values())+runtime['shared_setup_seconds']
                    +runtime['model_module_setup_seconds']+math.fsum(calls[k]['seconds'] for k in expected if k != 'model_load'))
        require(disjoint <= worker['wall_seconds']+1e-8, 'nonoverlapping qualification work within worker time')
        require(math.fsum(r['excluded_io_seconds'] for r in setup.values()) <= worker['artifact_io_seconds']+1e-8,
                'only measured setup I/O excluded')
        return {**result, 'audit_version': VERSION, 'agreement': True, 'comparisons': c.count,
                'maximum_prediction_difference': self.maximum_prediction_error, 'actual_work': expected,
                'saved_checkpoint_readout_calls': 780, 'saved_network_rows': 12480,
                'costs': {'runtime': runtime, 'inference_setup': setup, 'preparation_seconds': seconds,
                          'disjoint_accounted_seconds': disjoint, 'worker_seconds': worker['wall_seconds']}, 'scope': SCOPE}


    def authenticate(self):
        a, c = self.args, self.c
        require_threads()
        for path in (a.plan, a.run, a.terminal, a.output):
            require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'absolute nonsymlink paths')
        for path, pin in ((a.plan, a.plan_sha256), (a.run/'receipt.json', a.receipt_sha256), (a.terminal, a.terminal_sha256)):
            c.equal(B.digest(path, self.check)['sha256'], pin, 'external identity before decoding')
        plan, worker, terminal = B.read(a.plan), B.read(a.run/'receipt.json'), B.read(a.terminal)
        require(plan['version'] == 'otto-spatial-control-v1' and plan['mode'] == a.mode
                and plan['status'] == 'frozen_before_execution', 'prospective matching mode')
        c.equal(plan['independent_audit_limits'], self.limits, 'prospective audit allocation')
        c.equal(plan['audit_serialization_projection'], journal_projection(a.mode), 'prospective independent audit byte allocation')
        c.equal(plan['audit_study_serialization_projection'], journal_projection('study'), 'full-study audit sizing before qualification')
        for name, pin in plan['sources'].items():
            path = ROOT/name
            require(path.resolve().is_relative_to(ROOT), 'repository source closure')
            c.equal(B.digest(path, self.check)['sha256'], pin, 'unchanged source: '+name)
        for name in (RUNNER, 'scripts/audit_otto_spatial_control.py', 'tests/test_audit_otto_spatial_control.py'):
            c.equal(B.digest(ROOT/name, self.check)['sha256'], plan['sources'][name], 'prospectively bound new source')
        for name, pin in {HELPER: HELPER_PIN, MODEL: MODEL_PIN}.items():
            c.equal(plan['sources'][name], pin, 'same frozen independent arithmetic')
        require(worker['status'] == 'completed' and worker['version'] == 'otto-spatial-control-v1'
                and worker['mode'] == a.mode and worker['pending'] == []
                and worker['external_model_calls'] == 0 and worker['completed_fits'] == 0,
                'completed no-training worker')
        for key in ('sources', 'inputs', 'limits'):
            c.equal(worker[key], plan[key], 'worker frozen '+key)
        c.equal(worker['plan_sha256'], a.plan_sha256, 'worker plan pin')
        names = payload_names(a.mode)
        c.equal(set(worker['files']), names, 'exact payload membership')
        c.equal({p.name for p in a.run.iterdir()}, names | {'receipt.json'}, 'closed successful directory')
        for name in names:
            c.equal(B.digest(a.run/name, self.check), worker['files'][name], 'closed payload before arrays')
        c.equal(B.read(a.run/'serialization-projection.json'), plan['serialization_projection'], 'same frozen producer byte projection')
        started = B.read(a.run/'started.json')
        request, launch = started['request'], started['launch']
        c.equal(request, {'plan': str(a.plan), 'plan_sha256': a.plan_sha256, 'output': str(a.run),
                         'supervision': request['supervision']}, 'original worker request')
        c.equal(B.digest(Path(request['supervision']), self.check)['sha256'], worker['supervision_sha256'], 'original launch file pin')
        c.equal(B.read(Path(request['supervision'])), launch, 'original launch contents')
        require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
                and terminal['group_absent'] is True and terminal['cleanup']['group_absent'] is True
                and terminal['cleanup']['reaped'] is True and terminal['cleanup']['errors'] == []
                and terminal['error'] is None and terminal['clock_error'] is None and terminal['timing_available'] is True,
                'successful parent before scientific decode')
        for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend',
                    'cap_seconds', 'clock_source_sha256', 'watchdog_sha256'):
            c.equal(terminal[key], launch[key], 'same closed parent')
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        require(len(command) == 10, 'exact original CLI fields')
        c.equal(command[:2], [plan['python_executable'], str(ROOT/RUNNER)], 'actual source and interpreter')
        c.equal(dict(zip(command[2::2], command[3::2], strict=True)),
                {f'--{k.replace("_", "-")}': v for k, v in request.items()}, 'original CLI bindings')
        cap = plan['limits']['native_seconds']
        require(launch['cap_seconds'] == cap and launch['pid'] == launch['pgid'] and launch['parent_pid'] != launch['pid']
                and launch['cwd'] == str(ROOT) and launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
                and worker['clock_backend'] == launch['clock_backend'] and launch['deadline_ns'] == launch['started_ns']+cap*10**9
                and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns'],
                'native timing enclosure')
        c.equal(started['started_ns'], worker['started_ns'], 'worker clock origin')
        c.equal(worker['wall_seconds'], (worker['finished_ns']-worker['started_ns'])/1e9, 'worker elapsed')
        c.equal(terminal['elapsed_ns'], terminal['finished_ns']-terminal['started_ns'], 'parent elapsed')
        c.equal(terminal['wall_seconds'], terminal['elapsed_ns']/1e9, 'parent wall interval')
        c.equal(launch['clock_source_sha256'], B.CLOCK_PIN, 'qualified native clock')
        c.equal(launch['watchdog_sha256'], plan['sources']['scripts/supervise_dialogue_observation_v2.py'], 'qualified supervisor')
        require(type(worker['peak_rss_bytes']) is int and 0 < worker['peak_rss_bytes'] <= plan['limits']['rss_bytes']
                and sum(p.stat().st_size for p in a.run.iterdir()) <= plan['limits']['output_bytes'], 'recorded resource limits')
        # Sole producer use authenticates complete old inputs and runtime. No
        # producer preparation, prediction, branch or aggregate is imported here.
        producer = B.load(ROOT/RUNNER, '_spatial_control_lineage_only')
        c.equal(producer.authenticate(argparse.Namespace(plan=a.plan, plan_sha256=a.plan_sha256)), plan,
                'closed prior spatial study, mode prerequisites and runtime')
        self.receipt.update(plan_sha256=a.plan_sha256, worker_sha256=a.receipt_sha256,
                            terminal_sha256=a.terminal_sha256, producer_source_sha256=plan['sources'][RUNNER])
        return plan, worker


    def episode(self, row, identity, events):
        np, c = self.np, self.c
        regime, seed, hit, arm, block = identity
        episode_id = f'eval:{regime}:{seed}:{arm}'
        canonical = {'episode_id': episode_id, 'stage': 'eval', 'regime': regime, 'seed': seed,
                     'initial_hit': hit, 'arm': arm, 'block': block}
        c.equal({k: row[k] for k in canonical}, canonical, 'all1152 evaluation identities')
        steps = row['steps']
        require(type(steps) is int and 1 <= steps <= HORIZON and type(row['found']) is bool
                and (row['found'] or steps == HORIZON), 'complete found/censored duration')
        context = {'phase': 'eval', 'episode': episode_id, 'step': 0}
        self.work.call('native_reset', context)
        public = B.packet([26, 26], hit, False, 0)
        source = row['source_evaluation_only']
        require(len(source) == 2 and all(type(v) is int and 0 <= v < 53 for v in source)
                and source != [26, 26], 'evaluator source geometry')
        probability = B.posterior(np.ones((53, 53), np.float64)/2808, public, self.kernels[regime], np)
        state = E.A.A.posterior_record(probability)
        c.equal(next(events), {'kind': 'reset', **canonical, 'public': public, 'posterior_after': state,
                               'source_evaluation_only': source}, 'evaluation initial public state')
        pair = (regime, seed)
        if pair in self.sources:
            c.equal((source, public), self.sources[pair], 'paired sampled source and reset')
        else:
            self.sources[pair] = (source, public)
        draws = iter(row['draws_evaluation_only'])
        first = next(draws)
        c.equal((first['channel'], first['index'], first['selected_index']), ('source', 0, 53*source[0]+source[1]), 'source draw identity')
        learned = arm != 'analytic_inbounds'
        totals = {key: [] for key in ('choose_seconds', 'update_seconds', 'environment_seconds', 'choose_instrumented_seconds', 'choose_excluded_io_seconds')}
        positions, zero = [], 0
        for step in range(1, steps+1):
            self.check()
            ctx = {**context, 'step': step}
            operation = self.work.call('value_forward' if learned else 'analytic_choose', ctx)
            native = self.work.call('native_step', ctx)
            event = next(events)
            c.equal((event['kind'], event['episode_id'], event['step']), ('step', episode_id, step), 'every ordered evaluation decision')
            allowed = public['valid_actions']
            c.equal(event['allowed_actions'], allowed, 'public-only action mask')
            action = E.choice(event['costs'], allowed, learned, np)
            c.equal(event['action'], action, 'exact first eligible recorded-cost action')
            if learned:
                _, z, successors, raw, weight = branches(probability, public['position'], self.kernels[regime], np)
                c.equal(event['raw_masses'], raw.tolist(), 'all sixteen independent raw branch masses')
                c.equal(event['weights'], weight.tolist(), 'all sixteen independent exact floors')
                self.readout_context = (self.episode_index, step)
                value = 64*self.predicted(z, successors, REGIMES[regime], arm)
                rebuilt = E.costs(value, weight, np)
                self.arrays_close(np.asarray(event['values'], np.float64), value, 'independent physical branch values')
                self.arrays_close(np.asarray(event['costs'], np.float64), rebuilt, 'independent explicit four costs')
                c.equal(action, E.choice(rebuilt.tolist(), allowed, True, np), 'independent exact selected action; no tie exemption')
            else:
                c.equal((event['raw_masses'], event['weights'], event['values']), (None, None, None), 'analytic has no neural telemetry')
                rebuilt = E.analytic_scores(probability, public['position'], self.kernels[regime], np)
                for i in allowed:
                    c.close(event['costs'][i], rebuilt[i], 'independent analytic objective')
                c.equal(action, E.choice(rebuilt, allowed, False, np), 'independent analytic action')
            positions.append(tuple(public['position']))
            zero += state['mass'] == 0
            target = B.move(public['position'], action)
            found = target == source
            require(target != public['position'] and (not found or step == steps), 'inbounds motion and immediate found stopping')
            observed = event['public']['hit']
            require(type(observed) is int and (observed == -2 if found else 0 <= observed < 4), 'hit category or terminal sentinel')
            after = B.packet(target, observed, found, step)
            c.equal(event['public'], after, 'independent public transition')
            c.equal(event['native_p_end'], float(found), 'sampled native termination')
            c.equal(event['posterior_before'], state, 'before-action public posterior')
            probability = B.posterior(probability, after, self.kernels[regime], np)
            state = E.A.A.posterior_record(probability)
            c.equal(event['posterior_after'], state, 'all updates including final found/censor')
            if not found:
                draw = next(draws)
                c.equal((draw['channel'], draw['index'], draw['selected_index']), ('hit', step-1, observed), 'chronological public hit draw')
            for key, values in totals.items():
                value = event[key]
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'finite nonnegative operation timing')
                values.append(value)
            c.close(event['environment_seconds'], native['seconds'], 'environment operation cost')
            c.close(event['choose_seconds'], event['choose_instrumented_seconds']-event['choose_excluded_io_seconds'], 'only recorded I/O excluded')
            require(event['choose_seconds']+1e-9 >= operation['seconds'], 'complete branch/feature/readout/mask duration encloses scalar operation')
            public = after
        B.exhausted(draws, 'episode draw sequence')
        for draw in row['draws_evaluation_only']:
            require(0 <= draw['uniform'] < 1 and math.isfinite(draw['cdf_mass']) and abs(draw['cdf_mass']-1) < 1e-10, 'finite recorded random draw')
            key = (*pair, draw['channel'], draw['index'])
            if key in self.uniforms:
                c.equal(draw['uniform'], self.uniforms[key], 'paired channel-index uniform witness')
            else:
                self.uniforms[key] = draw['uniform']
        c.equal(row['final_public'], public, 'complete final packet')
        c.equal((row['found'], row['updates'], row['blocked_steps'], row['final_update_assimilated']),
                (public['done'], steps, 0, True), 'all complete outcomes/updates retained')
        tail = positions[-256:]
        diagnostic = (zero, state['mass'], sum(tail[i] == tail[i-2] for i in range(2, len(tail))),
                      max(0, len(tail)-2), len(set(positions)))
        c.equal(tuple(row[k] for k in ('zero_mass_decisions', 'final_posterior_mass', 'last256_lag2_matches', 'last256_lag2_pairs', 'distinct_preaction_positions')),
                diagnostic, 'descriptive mass and spatial repetition only')
        for key, values in totals.items():
            c.close(row[key], math.fsum(values), 'summed full episode operation costs')
        c.close(row['setup_allocation_seconds'], 0., 'physical episode before setup allocation')
        c.close(row['controller_seconds'], math.fsum(row[k] for k in TIMES), 'complete physical controller cost')
        require(math.isfinite(row['init_seconds']) and row['init_seconds'] >= 0, 'finite controller initialization')
        c.equal(row['state_bytes'], 22472, 'full public belief state storage')
        storage = row['storage']
        c.equal(set(storage), {'public_actor', 'head', 'branch_workspace'}, 'all retained storage components')
        c.equal({k: v for k, v in storage['public_actor'].items() if k != 'scope'},
                {'immutable_array_bytes': 366368+91592, 'mutable_array_bytes': 22472,
                 'immutable_arrays': {'observation_kernel': 366368, 'manhattan_distance_table': 91592}}, 'including inherited unused distance table')
        c.equal({k: v for k, v in storage['head'].items() if k != 'scope'} if learned else storage['head'],
                {k: v for k, v in self.heads[arm].storage_bytes().items() if k != 'scope'} if learned else None, 'actual float64 runtime head storage')
        c.equal(storage['branch_workspace'], '16*105*105 float64 centered u and z plus finite branch/features workspace, transient', 'transient workspace disclosure')
        return row


    def study(self, plan, worker, runtime):
        c = self.c
        require(worker['completed_episodes'] == 1152 and worker['parity_passed'] is True, 'complete autonomous cohort')
        native = B.read(self.args.run/'native-setup.json')
        c.equal(native['native_resets'], 3, 'exact template resets')
        require(len(native['checks']) == 3, 'all three native kernel checks')
        for i, (regime, row) in enumerate(zip(REGIMES, native['checks'], strict=True)):
            self.work.call('native_reset', {'phase': 'native_setup', 'regime': regime})
            c.equal((row['regime'], row['template_seed'], row['cached_kernel_exact']), (regime, 16500001+i, True), 'same kernel/template witness')
            hit = row['initial_public']['hit']
            require(type(hit) is int and 1 <= hit <= 3, 'positive template initial hit')
            c.equal(row['initial_public'], B.packet([26, 26], hit, False, 0), 'public template packet')
        setup = self.inference_setup(plan)
        episodes = iter(B.lines(self.args.run/'eval-episodes.jsonl', self.check))
        allocated = iter(B.lines(self.args.run/'evaluation.jsonl', self.check))
        events = iter(B.lines(self.args.run/'eval-transitions.jsonl', self.check))
        self.sources, self.uniforms, rows = {}, {}, []
        for self.episode_index, identity in enumerate(evaluation_order()):
            row = self.episode(next(episodes), identity, events)
            arm = row['arm']
            allocation = setup[arm]['seconds']/72+runtime['model_module_setup_seconds']/1080 if arm in setup else 0.
            result = {**row, 'setup_allocation_seconds': allocation, 'controller_seconds': row['controller_seconds']+allocation}
            c.equal(next(allocated), result, 'complete allocated controller cost')
            rows.append({k: v for k, v in result.items() if k != 'draws_evaluation_only'})
        for stream, label in ((episodes, 'episodes'), (allocated, 'allocated episodes'), (events, 'all public transitions')):
            B.exhausted(stream, 'complete '+label)
        expected = {'model_load': 15, 'native_reset': 1155, 'native_step': sum(r['steps'] for r in rows),
                    'analytic_choose': sum(r['steps'] for r in rows if r['arm'] == 'analytic_inbounds'),
                    'value_forward': sum(r['steps'] for r in rows if r['arm'] != 'analytic_inbounds')}
        calls = self.finish_work(worker, expected)
        require(expected['native_step'] <= plan['limits']['native_steps'] and expected['native_reset'] <= plan['limits']['native_resets']
                and expected['value_forward'] <= plan['limits']['value_forward'], 'prospective native and learned operation caps')
        c.equal((self.receipt['saved_checkpoint_readout_calls'], self.receipt['saved_network_rows']),
                (expected['value_forward'], 16*expected['value_forward']), 'one independent16-row readout per learned decision')
        result = aggregate(rows, self.mixtures)
        saved = B.read(self.args.run/'summary.json')
        for key, value in result.items():
            c.tree(saved[key], value, 'independent summary/'+key)
        c.equal(worker['pilot_continuation'], result['pilot_continuation'], 'all66 unchanged scientific decisions')
        c.tree(saved['calls'], calls, 'summary actual work')
        c.equal(saved['inference_setup'], setup, 'summary paid head setup')
        for key in ('shared_setup_seconds', 'model_module_setup_seconds'):
            c.equal(saved[key], runtime[key], 'summary setup scope')
        c.equal((saved['paired_source_cases'], saved['paired_uniforms']), (72, len(self.uniforms)), 'all paired source and random channels')
        scenarios = amortization(result, setup, self.training_reference['training_costs'],
                                self.training_reference['preparation_seconds'], runtime['model_module_setup_seconds'])
        c.tree(saved['amortization']['scenarios'], scenarios, 'H1/H100/H10000 independent accounting')
        training_reference = {k: self.training_reference[k] for k in ('training_costs', 'diagnostic_costs', 'preparation_seconds')}
        training_reference['worker_seconds'] = B.read(self.spatial_run/'receipt.json')['wall_seconds']
        c.equal(saved['amortization']['training_reference'], training_reference, 'already-paid fitting intervals unchanged')
        disjoint = (math.fsum(r['controller_seconds']+r['environment_seconds'] for r in rows)
                    +runtime['shared_setup_seconds']+calls['native_reset']['seconds'])
        require(disjoint <= worker['wall_seconds']+1e-8, 'complete nonoverlapping runtime accounting')
        excluded = math.fsum(r['choose_excluded_io_seconds'] for r in rows)+math.fsum(r['excluded_io_seconds'] for r in setup.values())
        require(excluded <= worker['artifact_io_seconds']+1e-8, 'only measured artifact I/O excluded')
        return {**result, 'audit_version': VERSION, 'agreement': True, 'amortization': scenarios,
                'comparisons': c.count, 'maximum_scalar_difference': c.maximum_difference,
                'maximum_prediction_difference': self.maximum_prediction_error, 'actual_work': expected,
                'saved_checkpoint_readout_calls': self.receipt['saved_checkpoint_readout_calls'],
                'saved_network_rows': self.receipt['saved_network_rows'],
                'condition_counts': {'competence': 18, 'improvement': 48, 'control_competence_descriptive': 72},
                'costs': {'training_reference': training_reference, 'runtime': runtime, 'inference_setup': setup,
                          'disjoint_accounted_seconds': disjoint, 'worker_seconds': worker['wall_seconds']}, 'scope': SCOPE}

    def compute(self, plan, worker):
        self.numerical_inputs(plan)
        runtime = B.read(self.args.run/'runtime.json')
        self.c.equal((runtime['python'], runtime['executable']), (sys.version, plan['python_executable']), 'recorded numerical interpreter')
        self.c.equal((runtime['torch_threads'], runtime['torch_interop_threads'], runtime['module_allocation_episodes']),
                     (1, 1, 0 if self.args.mode == 'qualify' else 1080), 'CPU1 and module allocation')
        self.c.equal(runtime['environment'], {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}, 'single-thread numerical settings')
        for key in ('shared_setup_seconds', 'model_module_setup_seconds'):
            E.finite(runtime[key], 'finite runtime setup')
        self.begin_work()
        return self.qualification(plan, worker, runtime) if self.args.mode == 'qualify' else self.study(plan, worker, runtime)

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists() and not any(p.is_symlink() for p in self.out.parents),
                'exclusive nonsymlink audit output')
        self.out.mkdir(parents=True, exist_ok=False)
        previous_alarm = signal.getsignal(signal.SIGALRM)
        try:
            require(B.digest(ROOT/B.CLOCK)['sha256'] == B.CLOCK_PIN, 'qualified suspend-inclusive clock')
            self.clock = B.load(ROOT/B.CLOCK, '_spatial_control_audit_clock').SuspendClock()
            self.start = self.clock.now_ns()
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('saved spatial audit deadline')))
            signal.setitimer(signal.ITIMER_REAL, self.limits['native_seconds'])
            self.receipt.update(source=B.digest(Path(__file__), self.check), independent_helper_sha256=HELPER_PIN,
                                qualified_shared_numpy_sha256=MODEL_PIN, environment=THREADS)
            B.write(self.out/'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()},
                    'limits': self.limits, 'clock_backend': self.clock.backend, 'started_ns': self.start,
                    'readout_journal_schema': {'attempt': ['call', 0, 'arm_index', 'episode_or_tuple_index', 'step', 16],
                                             'return': ['call', 1, 'elapsed_nanoseconds']},
                    'journal_projection': journal_projection(self.args.mode), 'environment': THREADS})
            plan, worker = self.authenticate()
            result = self.compute(plan, worker)
            require(self.receipt['readout_attempts'] == self.receipt['readout_returns']
                    == self.receipt['saved_checkpoint_readout_calls'] and self.receipt['pending'] is None,
                    'all real saved-model calls returned')
            self.authenticate()
            result.update(readout_attempts=self.receipt['readout_attempts'],
                          readout_returns=self.receipt['readout_returns'],
                          readout_seconds=self.receipt['readout_seconds'],
                          journal_projection=journal_projection(self.args.mode))
            B.write(self.out/'summary.json', result)
            self.check()
            finish = self.clock.now_ns()
            self.receipt.update(status='completed', agreement=True, clock_backend=self.clock.backend,
                                started_ns=self.start, finished_ns=finish, wall_seconds=(finish-self.start)/1e9,
                                comparisons=self.c.count, files={p.name: B.digest(p, self.check) for p in self.out.iterdir()})
            B.write(self.out/'receipt.json', self.receipt)
            self.check()
            return self.receipt
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            self.receipt.update(status='failed', agreement=False, error=repr(error), traceback=traceback.format_exc())
            try:
                self.receipt['wall_seconds'] = None if self.clock is None or self.start is None else (self.clock.now_ns()-self.start)/1e9
            except BaseException as clock_error:  # noqa: BLE001 - Retain the primary failure when the terminal clock also fails.
                self.receipt.update(wall_seconds=None, terminal_clock_error=repr(clock_error))
            try:
                if (self.out/'receipt.json').exists():
                    (self.out/'receipt.json').rename(self.out/'invalid-completed-receipt.json')
                B.write(self.out/'receipt.json', self.receipt)
                B.write(self.out/'failed.json', {'status': 'failed', 'error': repr(error)})
            except BaseException as secondary:  # noqa: BLE001 - Preserve the primary failure.
                error.add_note(f'Audit failure publication: {secondary!r}')
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=tuple(LIMITS), required=True)
    for flag in ('plan', 'run', 'terminal', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    for flag in ('plan-sha256', 'receipt-sha256', 'terminal-sha256'):
        parser.add_argument('--'+flag, required=True)
    Audit(parser.parse_args()).execute()
