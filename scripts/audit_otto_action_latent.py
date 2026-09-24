"""Independently score saved action-gap predictions after original process closure.

Only DEV labels and saved predictions are decoded. Scalar proper scores, legal
action choices, case aggregation and target fingerprints are reconstructed here,
without importing the producer scorer, learner, model, simulator or ridge solver.
The fixed comparison function may consume these independently rebuilt reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import otto_action_latent_common as c

VERSION = 'otto-action-latent-saved-audit-v1'
FAMILIES = ('action_recurrent', 'action_blind', 'direct_horizon', 'ridge')
CONDITIONS = ('gap', 'normal')
HORIZONS = tuple(range(1, 9))
GROUPS = {'all': HORIZONS, 'short': HORIZONS[:4], 'long': HORIZONS[4:]}
DATA_FIELDS = {'prefix', 'prefix_lengths', 'actions', 'continuation', 'outcomes',
               'raw_costs', 'legal', 'case_ids', 'regimes'}


def close_equal(actual, expected, name='saved report'):
    if isinstance(expected, dict):
        c.require(type(actual) is dict and set(actual) == set(expected), name + ': exact fields')
        for key in expected:
            close_equal(actual[key], expected[key], name + '.' + key)
    elif isinstance(expected, list):
        c.require(type(actual) is list and len(actual) == len(expected), name + ': complete list')
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            close_equal(left, right, f'{name}[{i}]')
    elif type(expected) is float:
        c.require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                  and abs(actual - expected) <= 1e-10 + 1e-10 * abs(expected), name + ': finite scalar agreement')
    else:
        c.require(type(actual) is type(expected) and actual == expected, name + ': exact identity')


def scalar_report(np, dataset, outcome_predictions, cost_predictions, *, family, fit_seed, condition, check=lambda: None):
    """Independent scalar arithmetic on supplied fabricated or admitted arrays."""
    target, raw, legal = (dataset[k] for k in ('outcomes', 'raw_costs', 'legal'))
    c.require(isinstance(target, np.ndarray) and target.ndim == 2 and target.shape[1] == 8
              and len(target) > 0 and target.dtype.kind in 'iu', 'integer eight-step outcomes')
    n = len(target)
    c.require(bool(((target >= 0) & (target <= 4)).all()), 'five known target classes')
    c.require(isinstance(legal, np.ndarray) and legal.shape == (n, 8, 4) and legal.dtype == np.bool_, 'legal Boolean mask')
    for array, shape, label in ((raw, (n, 8, 4), 'teacher costs'),
                                (cost_predictions, (n, 8, 4), 'cost predictions'),
                                (outcome_predictions, (n, 8, 5), 'outcome predictions')):
        c.require(isinstance(array, np.ndarray) and array.shape == shape and array.dtype.kind in 'fiu'
                  and bool(np.isfinite(array).all()), 'finite complete ' + label)
    ids, regimes = list(dataset['case_ids']), list(dataset['regimes'])
    c.require(len(ids) == len(regimes) == n and all(isinstance(v, str) and v.strip() for v in ids + regimes),
              'complete case and regime strings')
    ids, regimes = list(map(str, ids)), list(map(str, regimes))
    c.require(all(len({regimes[i] for i in range(n) if ids[i] == case}) == 1 for case in set(ids)), 'one regime per originating case')
    c.require(family in FAMILIES and type(fit_seed) is int and condition in CONDITIONS, 'declared view identity')
    logs, briers, gaps, actions = [], [], [], []
    for i in range(n):
        check()
        log_row, brier_row, gap_row, action_row = [], [], [], []
        found = False
        for t in range(8):
            check()
            label = int(target[i, t])
            c.require(not found or label == 4, 'absorbing target suffix')
            found = label == 4
            allowed = [a for a in range(4) if bool(legal[i, t, a])]
            c.require(bool(allowed) is (not found), 'nonterminal-only legal decisions')
            values = [float(v) for v in outcome_predictions[i, t]]
            if family == 'ridge':
                c.require(all(0 <= v <= 1 for v in values), 'probability interval')
                total = math.fsum(values)
                c.require(abs(total - 1) <= 1e-12, 'unit probability row')
                probabilities = [v / total for v in values]
                c.require(probabilities[label] > 0, 'observed zero probability has infinite logarithmic loss')
                log_score = -math.log(probabilities[label])
            else:
                maximum = max(values)
                shifted = [v - maximum for v in values]
                c.require(all(math.isfinite(v) for v in shifted), 'finite shifted logits')
                exponential = [math.exp(v) for v in shifted]
                total = math.fsum(exponential)
                probabilities = [v / total for v in exponential]
                log_score = math.log(total) - shifted[label]
            brier = math.fsum((p - int(a == label)) ** 2 for a, p in enumerate(probabilities))
            c.require(math.isfinite(log_score) and log_score >= 0 and math.isfinite(brier), 'finite proper scores')
            if allowed:
                # Tuple ordering makes exact legal ties choose the smallest ID.
                chosen = min(allowed, key=lambda a: (float(cost_predictions[i, t, a]), a))
                gap = float(raw[i, t, chosen]) - min(float(raw[i, t, a]) for a in allowed)
                c.require(math.isfinite(gap) and gap >= 0, 'finite independent decision gap')
            else:
                chosen, gap = -1, None
            log_row.append(log_score)
            brier_row.append(brier)
            gap_row.append(gap)
            action_row.append(chosen)
        logs.append(log_row)
        briers.append(brier_row)
        gaps.append(gap_row)
        actions.append(action_row)

    def mean(values):
        result = math.fsum(v / len(values) for v in values)
        c.require(math.isfinite(result), 'finite aggregate arithmetic')
        return result

    def summary(indices, horizons):
        records = []
        for case in sorted({ids[i] for i in indices}):
            check()
            members = [i for i in indices if ids[i] == case]
            pairs = [(i, h - 1) for i in members for h in horizons]
            supported = [(i, t) for i, t in pairs if gaps[i][t] is not None]
            records.append({'case_id': case, 'blocks': len(members), 'outcome_rows': len(pairs),
                'decision_rows': len(supported), 'terminal_rows': len(pairs) - len(supported),
                'log_score': mean([logs[i][t] for i, t in pairs]),
                'brier': mean([briers[i][t] for i, t in pairs]),
                'decision_gap': mean([gaps[i][t] for i, t in supported]) if supported else None})
        supported = [r['decision_gap'] for r in records if r['decision_gap'] is not None]
        return {'horizons': list(horizons), 'declared_cases': len(records), 'blocks': len(indices),
            'outcome_rows': sum(r['outcome_rows'] for r in records),
            'terminal_rows': sum(r['terminal_rows'] for r in records),
            'decision_rows': sum(r['decision_rows'] for r in records), 'supported_cases': len(supported),
            'unsupported_case_ids': [r['case_id'] for r in records if r['decision_gap'] is None],
            'case_weighted_log_score': mean([r['log_score'] for r in records]),
            'case_weighted_brier': mean([r['brier'] for r in records]),
            'case_weighted_decision_gap': mean(supported) if supported else None,
            'full_case_denominator_gap': mean([r['decision_gap'] if r['decision_gap'] is not None else 0. for r in records]) if supported else None,
            'cases': records}

    def groups(horizons):
        return {'overall': summary(list(range(n)), horizons),
                'by_regime': {r: summary([i for i in range(n) if regimes[i] == r], horizons) for r in sorted(set(regimes))}}

    digest = hashlib.sha256(b'otto-action-latent-targets-v1\0')
    digest.update(target.astype('<i8').tobytes())
    digest.update(legal.tobytes())
    digest.update(raw.astype('<f8').tobytes())
    return {'version': 'otto-action-latent-metrics-v1', 'family': family, 'fit_seed': fit_seed,
        'condition': condition, 'prediction_kind': 'probabilities' if family == 'ridge' else 'logits',
        'horizons': list(HORIZONS), 'target_sha256': digest.hexdigest(),
        'identity_manifest': [{'block_index': i, 'case_id': ids[i], 'regime': regimes[i]} for i in range(n)],
        'per_horizon': {str(h): groups((h,)) for h in HORIZONS},
        'groups': {name: groups(horizons) for name, horizons in GROUPS.items()}, 'chosen_actions': actions,
        'scope': 'proper five-outcome prediction and teacher-action imitation on fixed precommitted blocks'}


def thresholds():
    cfg = c.CONFIG
    return {'long_log_relative_gain': cfg['long_logscore_improvement'],
        'long_gap_relative_gain': cfg['long_gap_improvement'],
        'normal_log_relative_tolerance': cfg['normal_relative_tolerance'],
        'normal_gap_relative_tolerance': cfg['normal_relative_tolerance'],
        'minimum_supported_cases': cfg['minimum_supported_cases']}


def verify_training(np, directory, receipt, collection_summary, summary, *, check=lambda: None):
    """Check recorded fit counts and deterministic orders, without replaying fits."""
    normalization = c.read(directory / 'normalization.json')
    normalization_fields = {'cost_scale', 'rms_squared_before_floor', 'variance_floor', 'source',
                            'teacher_units_divisor', 'dev_decodes'}
    c.require(set(normalization) == normalization_fields
              and normalization['source'] == 'TRAIN only, all-four centered, equal cases and surviving rows'
              and normalization['variance_floor'] == 1e-6 and normalization['teacher_units_divisor'] == 64.
              and type(normalization['dev_decodes']) is int and normalization['dev_decodes'] == 0,
              'fixed TRAIN-only normalization metadata')
    variance, scale = normalization['rms_squared_before_floor'], normalization['cost_scale']
    c.require(type(variance) is float and math.isfinite(variance) and variance >= 0
              and type(scale) is float and math.isfinite(scale) and scale > 0,
              'finite recorded variance and positive float32 scale')
    with np.errstate(over='raise', invalid='raise'):
        try:
            expected_scale = float(np.float32(np.sqrt(max(variance, 1e-6))))
        except FloatingPointError as error:
            raise ValueError('finite float32 normalization arithmetic') from error
    c.require(scale == expected_scale, 'scale matches recorded variance and fixed floor; variance is not recomputed')
    n = collection_summary['counts']['train']['lambda3']
    cfg = c.CONFIG
    c.require(type(n) is int and n >= cfg['min_train'] and collection_summary['counts']['train']['lambda4'] == 0,
              'recorded surviving TRAIN cohort')
    epochs, batch = cfg['epochs'], cfg['batch']
    expected = []
    neural = FAMILIES[:3]
    for i, seed in enumerate(cfg['fit_seeds']):
        expected.extend((family, seed) for family in neural[i:] + neural[:i])
    with (directory / 'fits.jsonl').open() as stream:
        fits = [json.loads(line) for line in stream]
    c.require(len(fits) == 9 and [(r['family'], r['seed']) for r in fits] == expected
              and summary['fits'] == fits, 'all nine ordered fit records')
    per_fit_updates = math.ceil(n / batch) * epochs
    for row in fits:
        check()
        c.require(row['epochs'] == epochs and row['updates'] == per_fit_updates and row['cases'] == n
                  and row['exposures'] == n * epochs and row['training_horizons'] == [1, 2, 3, 4]
                  and row['evaluation_decodes_so_far'] == 0, 'fixed complete TRAIN-only fit counts')
        for field in ('seconds', 'last_batch_loss', 'last_gradient_norm'):
            c.require(type(row[field]) in (int, float) and math.isfinite(row[field]) and row[field] >= 0,
                      'finite recorded fit scalar')
        checkpoint = f"{row['family']}-{row['seed']}.npz"
        c.require(row['checkpoint'] == receipt['files'][checkpoint], 'recorded checkpoint identity')
        parameters = row['parameters']
        count = {'action_recurrent': 8299, 'action_blind': 8299, 'direct_horizon': 8107}[row['family']]
        c.require(parameters['kind'] == row['family'] and parameters['hidden_dim'] == 28
                  and parameters['count'] == parameters['trainable_count'] == count
                  and parameters['cost_scale'] == scale and 'cost_scale' not in parameters['parameters']
                  and sum(v['count'] for v in parameters['parameters'].values()) == count,
                  'declared architecture parameter count and shared fixed normalization')
        c.require(isinstance(row['changed_tensors'], list) and row['changed_tensors']
                  and len(set(row['changed_tensors'])) == len(row['changed_tensors'])
                  and set(row['changed_tensors']) <= set(parameters['parameters']), 'changed tensor names retained')
    with (directory / 'training-orders.jsonl').open() as stream:
        for family, seed in expected:
            generator = np.random.Generator(np.random.PCG64(seed))
            for epoch in range(epochs):
                check()
                line = stream.readline()
                c.require(line.endswith('\n'), 'complete deterministic epoch record')
                row = json.loads(line)
                wanted = {'family': family, 'seed': seed, 'epoch': epoch, 'indices': generator.permutation(n).tolist()}
                c.require(row == wanted, 'same complete deterministic originating-case permutation')
        c.require(stream.read() == '', 'no extra training orders')
    ridge = c.read(directory / 'ridge-fit.json')
    c.require(ridge == summary['ridge_fit'] and ridge['fits'] == 1 and ridge['parameters'] == 336 * 9
              and ridge['training_horizons'] == [1, 2, 3, 4] and ridge['evaluation_decodes_so_far'] == 0
              and ridge['normal_equation_rows'] == 8 * n
              and type(ridge['seconds']) in (int, float) and math.isfinite(ridge['seconds']) and ridge['seconds'] >= 0,
              'one fixed TRAIN-only ridge fit')
    calls = {'optimizer_steps': 9 * per_fit_updates, 'fit_count': 9, 'dev_array_decodes': 1}
    c.require(summary['calls'] == receipt['calls'] == calls and summary['data_cases']['train'] == n,
              'recorded complete fit and evaluation counts')
    return {'fits_checked': 9, 'ridge_fits_checked': 1, 'normalization_records_checked': 1,
            'training_epochs_checked': 9 * epochs,
            'optimizer_steps_declared': calls['optimizer_steps'], 'training_case_exposures_declared': 9 * n * epochs}


def admit(run):
    """Authenticate both original closures before numerical imports or decoding."""
    collection, fitted = c.OUT / 'collection-01', c.OUT / 'fit-01'
    receipts = {}
    previous = None
    for phase, directory, terminal_name in (('collect', collection, 'collection-native-01.terminal.json'),
                                             ('fit', fitted, 'fit-native-01.terminal.json')):
        terminal_path = c.OUT / terminal_name
        receipt = c.closed(directory, terminal_path)
        terminal = c.read(terminal_path)
        c.require(receipt['version'] == c.VERSION and receipt['phase'] == phase
                  and receipt['plan_sha256'] == run.args.plan_sha256
                  and receipt['old_test_decodes'] == receipt['astra_calls'] == 0,
                  'same registered predecessor phase with no prohibited calls')
        c.require(previous is None or previous <= terminal['started_ns'], 'collection closes before fitting starts')
        previous = terminal['finished_ns']
        receipts[phase] = receipt
    c.require(previous <= run.launch['started_ns'], 'fit closes before independent audit starts')
    expected_fit = {'started.json', 'fits.jsonl', 'training-orders.jsonl', 'normalization.json', 'ridge.npz', 'ridge-fit.json',
                    'predictions.npz', 'reports.json', 'summary.json'} | {
                        f'{family}-{seed}.npz' for family in FAMILIES[:3] for seed in c.CONFIG['fit_seeds']}
    c.require({'dev.npz', 'summary.json'} <= set(receipts['collect']['files'])
              and set(receipts['fit']['files']) == expected_fit, 'exact closed fit evidence payloads')
    c.require(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax', 'mlx')), 'no model runtime in independent audit')
    return collection, fitted, receipts


def run_audit(run):
    collection, fitted, receipts = admit(run)
    import numpy as np

    from openjev.research import otto_action_latent_metrics as metrics

    counts = {'array_decodes': 0, 'views_checked': 0, 'model_calls': 0, 'optimizer_calls': 0,
              'solver_calls': 0, 'teacher_calls': 0, 'native_calls': 0, 'old_test_decodes': 0}
    run.receipt['audit_counts'] = counts
    collection_summary = c.read(collection / 'summary.json')
    fit_summary = c.read(fitted / 'summary.json')
    counts.update(verify_training(np, fitted, receipts['fit'], collection_summary, fit_summary, check=run.check))
    c.require(fit_summary['collection_receipt'] == c.desc(collection / 'receipt.json')
              and fit_summary['train_data'] == receipts['collect']['files']['train.npz']
              and fit_summary['dev_data'] == receipts['collect']['files']['dev.npz'], 'fit binds original collection inputs')

    def arrays(directory, name, phase):
        run.check()
        c.require(c.desc(directory / name) == receipts[phase]['files'][name], 'unchanged predecode payload')
        run.receipt['pending'] = {'decode': name}
        with np.load(directory / name, allow_pickle=False) as archive:
            result = {k: archive[k] for k in archive.files}
        counts['array_decodes'] += 1
        run.receipt['pending'] = None
        return result

    dataset = arrays(collection, 'dev.npz', 'collect')
    predictions = arrays(fitted, 'predictions.npz', 'fit')
    c.require(set(dataset) == DATA_FIELDS, 'exact complete DEV data fields')
    seeds = c.CONFIG['fit_seeds']
    specs = [(family, seed, condition) for family in FAMILIES for seed in seeds for condition in CONDITIONS]
    expected_keys = {f'{a}__{s}__{condition}__{field}' for a, s, condition in specs for field in ('outcome', 'cost')}
    c.require(set(predictions) == expected_keys and len(expected_keys) == 48, 'exact 24-view saved prediction roster')
    identities = [str(v) for v in dataset['case_ids']]
    regimes = [str(v) for v in dataset['regimes']]
    roster = {r['id']: r for r in run.plan['roster'] if r['split'] == 'dev'}
    c.require(len(identities) == len(set(identities)) == len(regimes)
              and all(i in roster and roster[i]['regime'] == r for i, r in zip(identities, regimes, strict=True)),
              'only registered originating DEV cases')
    for regime in ('lambda3', 'lambda4'):
        actual = regimes.count(regime)
        c.require(actual == collection_summary['counts']['dev'][regime] >= c.CONFIG['min_dev_per_regime'], 'surviving-prefix count matches original collection')
    c.require(fit_summary['data_cases']['dev'] == len(identities), 'same complete evaluation cohort')
    timings = fit_summary['prediction_times']
    c.require(len(timings) == 24 and {(r['family'], r['seed'], r['condition']) for r in timings} == set(specs),
              'complete saved inference cost roster')
    for row in timings:
        c.require(row['cases'] == len(identities) and type(row['seconds']) in (int, float)
                  and math.isfinite(row['seconds']) and row['seconds'] >= 0
                  and type(row['work']) is dict and all(type(v) is int and v >= 0 for v in row['work'].values()),
                  'finite complete recorded inference costs')
    saved = c.read(fitted / 'reports.json')
    c.require(set(saved) == {'reports', 'gate'} and len(saved['reports']) == 24, 'complete saved report and gate payload')
    saved_by = {(r['family'], r['fit_seed'], r['condition']): r for r in saved['reports']}
    c.require(len(saved_by) == 24 and set(saved_by) == set(specs), 'unique complete report roster')
    for condition in CONDITIONS:
        for field in ('outcome', 'cost'):
            reference = predictions[f'ridge__{seeds[0]}__{condition}__{field}']
            c.require(all(predictions[f'ridge__{seed}__{condition}__{field}'].dtype == reference.dtype
                          and predictions[f'ridge__{seed}__{condition}__{field}'].shape == reference.shape
                          and predictions[f'ridge__{seed}__{condition}__{field}'].tobytes() == reference.tobytes() for seed in seeds),
                      'single deterministic ridge fit disclosed as repeated views')
    reports = []
    for family, seed, condition in specs:
        run.check()
        run.receipt['pending'] = {'family': family, 'fit_seed': seed, 'condition': condition}
        prefix = f'{family}__{seed}__{condition}__'
        report = scalar_report(np, dataset, predictions[prefix + 'outcome'], predictions[prefix + 'cost'],
                               family=family, fit_seed=seed, condition=condition, check=run.check)
        close_equal(saved_by[family, seed, condition], report)
        reports.append(report)
        counts['views_checked'] += 1
        run.receipt['pending'] = None
    gate = metrics.evaluate_reports(reports, candidate=c.CONFIG['candidate'], controls=c.CONFIG['controls'],
                                    fit_seeds=seeds, regimes=('lambda3', 'lambda4'), thresholds=thresholds())
    close_equal(saved['gate'], gate, 'fixed gate')
    c.require(fit_summary['status'] == ('DEV_PASS' if gate['passed'] else 'DEV_FAIL'), 'original fixed candidate outcome')
    for phase, directory in (('collect', collection), ('fit', fitted)):
        for name in ('dev.npz', 'summary.json') if phase == 'collect' else (
                'predictions.npz', 'reports.json', 'fits.jsonl', 'training-orders.jsonl',
                'normalization.json', 'ridge-fit.json', 'summary.json'):
            c.require(c.desc(directory / name) == receipts[phase]['files'][name], 'audit input bytes unchanged')
    c.require(counts['array_decodes'] == 2 and counts['views_checked'] == 24, 'complete independent audit accounting')
    result = {'version': VERSION, 'agreement': True, 'technical_complete': False,
        'requires_original_supervisor_closure': True, 'counts': counts, 'reports': reports, 'gate': gate,
        'normalization': {'record': c.read(fitted / 'normalization.json'), 'train_variance_recomputed': False,
                          'scale_from_recorded_variance_checked': True, 'fit_metadata_scale_joins_checked': 9},
        'inputs': {'collection_receipt': c.desc(collection / 'receipt.json'),
                   'fit_receipt': c.desc(fitted / 'receipt.json')},
        'limitations': ['No model, fit, simulator or independent causal rollout is replayed.',
                        'Training orders and counts are checked as records; optimizer execution is not replayed.',
                        'TRAIN-derived variance is authenticated metadata, not independently recomputed from TRAIN arrays.',
                        'Three ridge views repeat one fit; repeated neural fits share cases.',
                        'This audit cannot admit old TEST, confirmation or new execution.']}
    run.check()
    c.write(run.out / 'audit.json', result)
    run.receipt['agreement'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'supervision', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    c.require(Path(sys.executable).absolute() == c.NUMERICAL, 'qualified numerical audit interpreter')
    run = c.Run(args, 'audit')
    try:
        run_audit(run)
        run.finish()
    except BaseException as error:
        run.finish(error)
        raise


if __name__ == '__main__':
    main()
