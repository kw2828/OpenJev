# SPDX-License-Identifier: GPL-3.0-only
"""Pin analysis v2: preserve homogeneous integer or string source-game identities."""
import math
import statistics

import numpy as np

from openjev.research.chess_pin_study import CORE_ARMS, UNION_ARMS, SEEDS

SPLITS = ('dev', 'shift')


def validate_arms(arms):
    arms = tuple(arms)
    extras = arms[len(CORE_ARMS):]
    allowed = tuple('union:'+a for a in UNION_ARMS)
    if (arms[:len(CORE_ARMS)] != CORE_ARMS or len(extras) > 2
            or len(set(arms)) != len(arms) or any(a not in allowed for a in extras)
            or tuple(a for a in allowed if a in extras) != extras):
        raise ValueError('Expected seven mandatory arms followed by at most two ordered union comparators')
    return arms


def comparisons(arms):
    return ('base', *(a for a in validate_arms(arms) if a != 'joint'))


def gate(metrics, arms):
    """Agreement thresholds are fractions, not percentage-point integers."""
    arms = validate_arms(arms)
    expected = {f'{a}-{s}' for a in ('base', *arms) for s in SEEDS}
    if set(metrics) != expected:
        raise ValueError('Missing or extra method/seed results')
    for panels in metrics.values():
        if set(panels) != set(SPLITS):
            raise ValueError('Incomplete panel membership')
        for r in panels.values():
            if (type(r['agreement']) not in (int, float) or not math.isfinite(r['agreement'])
                    or not 0 <= r['agreement'] <= 1 or r['examples'] != 2048
                    or not math.isfinite(r['target_nll']) or r['target_nll'] < 0):
                raise ValueError('Invalid complete-panel metrics')
    checks = []
    for split in SPLITS:
        for arm in comparisons(arms):
            gains = [metrics[f'joint-{s}'][split]['agreement']-metrics[f'{arm}-{s}'][split]['agreement'] for s in SEEDS]
            threshold = 0. if arm == 'base' else .01
            mean, floor = statistics.mean(gains), min(gains)
            checks.append({'split': split, 'comparator': arm, 'paired_seed_gains': gains,
                           'mean_gain': mean, 'minimum_paired_seed_gain': floor,
                           'required_mean_gain': threshold, 'required_seed_floor': -.005,
                           'passed': mean >= threshold and floor >= -.005})
    return checks


def validate_game_ids(game_ids):
    # JSON integer zero is a valid nominal identifier. Do not coerce IDs to strings,
    # and reject mixed types/bools to avoid Python equality merging False with zero.
    if not game_ids:
        raise ValueError('Expected nonempty source-game identities')
    kinds = {type(g) for g in game_ids}
    if (len(kinds) != 1 or not kinds.issubset({int, str})
            or any(type(g) is str and not g for g in game_ids)):
        raise ValueError('Expected homogeneous integer or nonempty string source-game identities')


def game_interval(differences, game_ids, split):
    """Pair seed-averaged correctness within source games; position-weight draws.

    This is conditional descriptive uncertainty on exposed development panels.
    It neither samples new training seeds nor corrects adaptive model selection.
    """
    if split not in SPLITS:
        raise ValueError('Expected a named panel')
    validate_game_ids(game_ids)
    values = np.asarray(differences, dtype=float)
    if (values.shape != (len(SEEDS), len(game_ids)) or not np.isfinite(values).all()
            or not np.isin(values, [-1, 0, 1]).all()):
        raise ValueError('Expected paired binary-correctness differences for every seed and root')
    ids = np.asarray(game_ids)
    groups = [np.flatnonzero(ids == game) for game in sorted(set(game_ids))]
    delta = values.mean(0)
    sums = np.asarray([delta[g].sum() for g in groups])
    sizes = np.asarray([len(g) for g in groups])
    draws = np.random.default_rng(996101+(split == 'shift')).integers(0, len(groups), (2000, len(groups)))
    estimate = sums[draws].sum(1)/sizes[draws].sum(1)
    return {'games': len(groups), 'roots': len(game_ids), 'draws': 2000,
            'seed': 996101+(split == 'shift'), 'point_gain': float(delta.mean()),
            'percentile95': np.quantile(estimate, [.025, .975]).tolist()}


def analyze(records, rows, arms):
    """Analyze complete saved predictions supplied by an authenticated caller.

    Keys are (arm, seed, split); row metadata includes id, game_id, target_uci.
    The caller must separately authenticate neural/native replay and receipts.
    """
    arms = validate_arms(arms)
    expected = {(a, s, p) for a in ('base', *arms) for s in SEEDS for p in SPLITS}
    if set(records) != expected or set(rows) != set(SPLITS):
        raise ValueError('Incomplete prediction or source-panel membership')
    metrics = {}
    for split in SPLITS:
        source = rows[split]
        if len(source) != 2048 or len({r['id'] for r in source}) != 2048:
            raise ValueError('Wrong panel size or duplicate source identity')
        for arm in ('base', *arms):
            for seed in SEEDS:
                values = records[arm, seed, split]
                if len(values) != 2048:
                    raise ValueError('Incomplete prediction vector')
                for i, (r, original) in enumerate(zip(values, source)):
                    if (r['index'] != i or r['id'] != original['id'] or r['game_id'] != original['game_id']
                            or type(r['correct']) is not bool or r['correct'] != (r['choice'] == original['target_uci'])
                            or not math.isfinite(r['target_nll']) or r['target_nll'] < 0):
                        raise ValueError('Prediction/source alignment or correctness differs')
                metrics.setdefault(f'{arm}-{seed}', {})[split] = {
                    'agreement': statistics.mean(r['correct'] for r in values),
                    'target_nll': statistics.mean(r['target_nll'] for r in values), 'examples': len(values)}
    checks = gate(metrics, arms)
    intervals = []
    for split in SPLITS:
        for arm in comparisons(arms):
            differences = [[int(x['correct'])-int(y['correct']) for x, y in
                            zip(records['joint', seed, split], records[arm, seed, split])] for seed in SEEDS]
            intervals.append({'split': split, 'comparator': arm,
                              **game_interval(differences, [r['game_id'] for r in rows[split]], split)})
    return {'metrics': metrics, 'gate_checks': checks,
            'quality_gate_passed': all(r['passed'] for r in checks), 'conditional_game_bootstrap': intervals,
            'prediction_records': sum(len(v) for v in records.values())}
