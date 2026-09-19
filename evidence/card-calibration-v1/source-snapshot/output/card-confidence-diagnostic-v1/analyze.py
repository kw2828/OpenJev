"""One saved-only, post hoc diagnostic; no models, environments, RNG or tuning.

This source must exist before invocation. Existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

COMPLETED = 'cc0d22c85fe8c9dadb36ce4b6c4fb047ad920c05a929752b1d0df2be94df22ba'
BINDINGS = 'bbd609e4d9542c6fb1e4bc51dd7144347a96f951c6a091a8a509d916c4d5f49d'
MODES = ('kalman', 'innovation_local', 'gated_delta')
CATEGORIES = ('unseen_selected', 'seen_hidden_top_rank_wrong',
              'seen_hidden_top_rank_correct_different_pending', 'currently_visible_selected')
FIELDS = ('raw_true_rank_probability', 'raw_pending_rank_probability', 'raw_max_probability',
          'picker_pending_rank_probability')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def bound(base, name, expected):
    path = (base / name).resolve()
    require(not Path(name).is_absolute() and path.is_relative_to(base.resolve()), 'Unsafe member path')
    require(sha(path) == expected, 'Bound bytes differ: ' + name)
    return path


def blank():
    return {'games': 0, 'successes': 0, 'native_steps': 0, 'second_card_decisions': 0,
            'raw_probability_entries': 0, 'raw_zero_probability_entries': 0, 'raw_nonfinite_entries': 0,
            'matching_second_cards': 0, 'queries': [], 'old_queries': [],
            'mismatches': {key: [] for key in CATEGORIES}}


def category(*, seen, visible, top, true_rank, pending_rank):
    require(true_rank != pending_rank, 'Only actual mismatches are classified')
    if visible:
        require(seen, 'Visible position must be publicly seen')
        return 'currently_visible_selected'
    if not seen:
        return 'unseen_selected'
    if top != true_rank:
        return 'seen_hidden_top_rank_wrong'
    return 'seen_hidden_top_rank_correct_different_pending'


def query_rows(raw, normalized, labels, eligible):
    """Return sufficient per-query values; callers pool exact denominators."""
    indices = np.flatnonzero(eligible)
    p = normalized[indices]
    y = labels[indices]
    chosen = p[np.arange(len(indices)), y]
    # Positive-mass input rows may assign exactly zero to a true rank. No epsilon clipping.
    positive = chosen > 0
    nll = np.zeros(len(indices), np.float64)
    nll[positive] = -np.log(chosen[positive])
    brier = np.sum(p * p, axis=1, dtype=np.float64) - 2 * chosen + 1
    return np.column_stack((np.argmax(raw[indices], axis=1) == y, np.max(raw[indices], axis=1),
                            np.max(p, axis=1), nll, ~positive, brier))


def episode(arrays, receipt):
    obs, actions, ranks, rewards, raw, picker, term, trunc = (
        arrays[key] for key in ('observations', 'actions', 'ranks', 'rewards',
                               'raw_probabilities', 'picker_probabilities', 'terminated', 'truncated'))
    n = len(actions)
    schema = {'observations': ((n + 1, 52), np.int64), 'actions': ((n,), np.int64),
              'ranks': ((n,), np.int64), 'rewards': ((n,), np.float64),
              'raw_probabilities': ((n, 52, 13), np.float64),
              'picker_probabilities': ((n, 52, 13), np.float64),
              'terminated': ((n,), np.bool_), 'truncated': ((n,), np.bool_)}
    require(set(arrays) == set(schema) and 1 <= n <= 104, 'Invalid episode schema/length')
    for key, (shape, dtype) in schema.items():
        require(arrays[key].shape == shape and arrays[key].dtype == dtype
                and np.isfinite(arrays[key]).all(), 'Invalid array ' + key)
    require(np.all(obs[0] == 13) and np.all((obs >= 0) & (obs <= 13)), 'Invalid public observations')
    require(np.all((raw >= 0) & (raw <= 1)) and np.allclose(raw.sum(2), 1, rtol=0, atol=1e-5),
            'Invalid saved softmax probability mass')
    labels = np.full(52, -1, np.int64)
    last_visible = np.full(52, -1, np.int64)
    matched, pending, mismatched_before = np.zeros(52, bool), None, set()
    result = blank()
    for t, action in enumerate(actions):
        a = int(action)
        current, after = obs[t], obs[t + 1]
        require(0 <= a < 52 and not matched[a] and a != pending, 'Invalid issued action')
        normalized = raw[t] / raw[t].sum(axis=1, dtype=np.float64)[:, None]
        actual_picker = normalized.copy()
        actual_picker[labels < 0] = 1 / 13
        visible = current != 13
        actual_picker[visible] = 0
        actual_picker[np.flatnonzero(visible), current[visible]] = 1
        require(np.array_equal(actual_picker, picker[t]), 'Saved C probability override differs')
        eligible = (labels >= 0) & ~visible
        ages = t - last_visible
        require(np.all(ages[eligible] >= 1), 'Invalid public visibility clock')
        result['queries'].append(query_rows(raw[t], normalized, labels, eligible))
        result['old_queries'].append(query_rows(raw[t], normalized, labels, eligible & (ages > 32)))
        expected_visible = matched.copy()
        expected_visible[a] = True
        if pending is not None:
            expected_visible[pending] = True
        require(np.array_equal(after != 13, expected_visible), 'Native returned visibility differs')
        known = expected_visible & (labels >= 0)
        require(np.array_equal(after[known], labels[known]) and ranks[t] == after[a], 'Public rank differs')
        true_rank = int(after[a])
        if pending is None:
            expected_reward, pending = 0., a
        else:
            result['second_card_decisions'] += 1
            pending_rank = int(current[pending])
            require(pending_rank == labels[pending] and pending_rank != 13, 'Pending card must be public')
            legal = np.flatnonzero(~matched)
            legal = legal[legal != pending]
            scores = actual_picker[legal, pending_rank]
            tied = legal[np.max(scores) - scores <= 1e-12]
            require(a == int(tied[0]), 'Selected second card differs from fixed C lexicographic tie rule')
            if true_rank == pending_rank:
                expected_reward = 2 / 52
                matched[[pending, a]] = True
                result['matching_second_cards'] += 1
            else:
                expected_reward = -2 / 104
                top = int(np.argmax(raw[t, a]))
                key = category(seen=bool(labels[a] >= 0), visible=bool(visible[a]), top=top,
                               true_rank=true_rank, pending_rank=pending_rank)
                position_pair = tuple(sorted((pending, a)))
                repeated = position_pair in mismatched_before
                mismatched_before.add(position_pair)
                result['mismatches'][key].append({
                    'raw_true_rank_probability': float(raw[t, a, true_rank]),
                    'raw_pending_rank_probability': float(raw[t, a, pending_rank]),
                    'raw_max_probability': float(np.max(raw[t, a])),
                    'picker_pending_rank_probability': float(actual_picker[a, pending_rank]),
                    'repeat_mismatch': repeated, 'raw_argmax_is_pending_rank': top == pending_rank})
            pending = None
        require(rewards[t] == expected_reward, 'Native reward differs')
        labels[expected_visible] = after[expected_visible]
        last_visible[expected_visible] = t + 1
        require(bool(term[t]) == bool(matched.all()) and bool(trunc[t]) == (t == 103), 'Terminal flags differ')
        require(bool(term[t] or trunc[t]) == (t == n - 1), 'Episode is incomplete or overrun')
    require(receipt['native_steps'] == n and receipt['return'] == math.fsum(rewards)
            and receipt['success'] == bool(matched.all()) and receipt['matched_pairs'] == int(matched.sum()) // 2,
            'Receipt outcome differs')
    result.update(games=1, successes=int(matched.all()), native_steps=n,
                  raw_probability_entries=int(raw.size), raw_zero_probability_entries=int(np.count_nonzero(raw == 0)))
    return result


def merge(target, source):
    for key in ('games', 'successes', 'native_steps', 'second_card_decisions', 'matching_second_cards',
                'raw_probability_entries', 'raw_zero_probability_entries', 'raw_nonfinite_entries'):
        target[key] += source[key]
    for key in ('queries', 'old_queries'):
        target[key].extend(source[key])
    for key in CATEGORIES:
        target['mismatches'][key].extend(source['mismatches'][key])


def distribution(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return {'count': 0, 'mean': None, 'min': None, 'q25': None, 'median': None, 'q75': None, 'max': None}
    return {'count': len(values), 'mean': math.fsum(values) / len(values), 'min': float(values.min()),
            'q25': float(np.quantile(values, .25)), 'median': float(np.median(values)),
            'q75': float(np.quantile(values, .75)), 'max': float(values.max())}


def query_summary(parts):
    values = np.concatenate(parts, axis=0) if parts else np.empty((0, 6))
    n = len(values)
    if not n:
        return {'eligible_queries': 0, 'correct': 0, 'accuracy': None, 'mean_raw_max_confidence': None,
                'mean_normalized_max_confidence': None, 'mean_nll_nats': None,
                'nll_status': 'no_eligible_queries', 'zero_true_probability_queries': 0, 'mean_brier_sum': None}
    means = [math.fsum(values[:, i]) / n for i in range(6)]
    zeros = int(values[:, 4].sum())
    return {'eligible_queries': n, 'correct': int(values[:, 0].sum()), 'accuracy': means[0],
            'mean_raw_max_confidence': means[1], 'mean_normalized_max_confidence': means[2],
            'accuracy_minus_raw_mean_confidence': means[0] - means[1],
            'mean_nll_nats': means[3] if not zeros else None,
            'nll_status': 'finite' if not zeros else 'positive_infinity_from_zero_true_probability',
            'zero_true_probability_queries': zeros, 'mean_brier_sum': means[5]}


def summarize(group):
    out = {key: group[key] for key in ('games', 'successes', 'native_steps', 'second_card_decisions', 'matching_second_cards',
                                     'raw_probability_entries', 'raw_zero_probability_entries', 'raw_nonfinite_entries')}
    total = sum(len(v) for v in group['mismatches'].values())
    require(total + out['matching_second_cards'] == out['second_card_decisions'], 'Nonexclusive mismatch partition')
    out['actual_mismatches'] = total
    out['all_eligible_seen_hidden_queries'] = query_summary(group['queries'])
    out['eligible_seen_hidden_age_gt32_queries'] = query_summary(group['old_queries'])
    out['mismatch_categories'] = {}
    for key, events in group['mismatches'].items():
        repeats = [event for event in events if event['repeat_mismatch']]
        n = len(events)
        out['mismatch_categories'][key] = {
            'mismatches': n, 'all_mismatches_denominator': total,
            'fraction_of_mismatches': n / total if total else None,
            'second_card_decisions_denominator': out['second_card_decisions'],
            'repeat_mismatches': len(repeats), 'repeat_fraction_within_category': len(repeats) / n if n else None,
            'raw_argmax_is_pending_rank': sum(event['raw_argmax_is_pending_rank'] for event in events),
            'probabilities': {field: distribution([event[field] for event in events]) for field in FIELDS},
            'repeat_mismatch_probabilities': {field: distribution([event[field] for event in repeats]) for field in FIELDS}}
    return out


def run(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(exist_ok=False)
    begin = time.monotonic()
    source_hash = sha(__file__)
    try:
        write(out / 'started.json', {'status': 'started', 'source_sha256': source_hash,
                                   'expected_execution_completed_sha256': COMPLETED,
                                   'scope': 'posthoc_saved_only_no_gate_change_no_tuning'})
        evaluation = root / 'output/card-controllers-v1/evaluation-01'
        require(sha(evaluation / 'completed.json') == COMPLETED, 'Unexpected completed evaluation')
        complete = read(evaluation / 'completed.json')
        require(complete['status'] == 'complete' and complete['episodes'] == 3840
                and len(complete['files']) == 7742, 'Incomplete authoritative run')
        actual = {str(p.relative_to(evaluation)) for p in evaluation.rglob('*') if p.is_file()}
        require(actual == set(complete['files']) | {'completed.json'}, 'Extra/missing run members')
        for name, item in complete['files'].items():
            path = bound(evaluation, name, item['sha256'])
            require(path.stat().st_size == item['bytes'], 'Payload size differs')
        bindings_path = bound(root, 'evidence/card-controllers-v1/source-bindings.json', BINDINGS)
        bindings = read(bindings_path)
        for name, digest in bindings['files'].items():
            bound(root, name, digest)
        protocol = read(bound(root, 'evidence/card-controllers-v1/protocol.json', complete['protocol_sha256']))
        inputs = read(bound(root, 'evidence/card-controllers-v1/inputs.json', complete['inputs_sha256']))
        old_ref = protocol['lineage']['old_bindings']
        old_bindings = read(bound(root, old_ref['path'], old_ref['sha256']))
        for name, digest in old_bindings['files'].items():
            bound(root, name, digest)
        groups = {mode: blank() for mode in MODES}
        fits, payloads, coverage = {}, {}, []
        for mode in MODES:
            for pair in range(3):
                name = f'{mode}-pair{pair}'
                fit = blank()
                for index, case in enumerate(inputs['evaluation']):
                    require(time.monotonic() - begin < 600, 'Saved-only diagnostic exceeded 600 seconds')
                    stem = evaluation / 'controllers' / name / 'C' / 'episodes' / f'{index:03d}'
                    receipt_path, arrays_path = stem.with_suffix('.json'), stem.with_suffix('.npz')
                    receipt = read(receipt_path)
                    require(receipt['status'] == 'complete' and receipt['controller'] == name and receipt['policy'] == 'C'
                            and receipt['index'] == index and receipt['seed'] == case['seed'], 'Episode identity differs')
                    require(receipt['layout_sha256'] == complete['layout_sha256_by_case'][str(index)], 'Paired layout differs')
                    require(sha(arrays_path) == receipt['npz_sha256'], 'Nested NPZ binding differs')
                    with np.load(arrays_path, allow_pickle=False) as archive:
                        arrays = {key: archive[key] for key in archive.files}
                    data = episode(arrays, receipt)
                    merge(fit, data)
                    coverage.append({'fit': name, 'case_index': index, 'native_steps': data['native_steps'],
                                     'success': bool(data['successes'])})
                    for path in (receipt_path, arrays_path):
                        rel = str(path.relative_to(evaluation))
                        payloads[rel] = complete['files'][rel]
                require(fit['games'] == 64, 'Incomplete fit coverage')
                merge(groups[mode], fit)
                fits[name] = summarize(fit)
        families = {mode: summarize(group) for mode, group in groups.items()}
        require(len(coverage) == 576 and all(g['games'] == 192 for g in groups.values()), 'Incomplete coverage')
        require(sha(__file__) == source_hash and sha(evaluation / 'completed.json') == COMPLETED, 'Boundary source changed')
        for name, digest in bindings['files'].items():
            bound(root, name, digest)
        result = {'status': 'complete', 'scope': 'exploratory_saved_C_histories_no_gate_change',
                  'source_sha256': source_hash, 'execution_completed_sha256': COMPLETED,
                  'source_bindings_sha256': BINDINGS, 'old_source_bindings_sha256': old_ref['sha256'],
                  'protocol_sha256': complete['protocol_sha256'], 'inputs_sha256': complete['inputs_sha256'],
                  'authenticated_run_payloads': 7742, 'analyzed_episode_payloads': payloads,
                  'families': families, 'fits': fits, 'coverage': coverage,
                  'definitions': {
                      'mismatch_partition': 'Second-card actual mismatches only; unseen, seen-hidden wrong top rank, seen-hidden correct top rank differing from pending rank, or currently visible selected.',
                      'labels': 'Seen-hidden queries use only frames through the pre-action observation. Unseen selected true-rank probabilities use its immediately returned public reveal only for outcome diagnosis; they do not enter recall metrics.',
                      'query_scope': 'Every previously public, currently hidden position before every actual action, including repeated queries and all successful/failed games. No hidden deck or future frame supplies a memory label.',
                      'age': 'Pre-action index minus latest public visibility index; old means strictly greater than 32 actions. Not necessarily last model-write age.',
                      'probabilities': 'Raw selected probabilities are unmodified saved softmax values. Proper NLL/Brier uses fixed float64 division by each saved row sum, matching C normalization before public overrides. No fitted temperature or epsilon clipping.',
                      'nll': 'Mean negative natural logarithm of normalized true-rank probability. Any zero true probability gives positive-infinite NLL, explicitly recorded with JSON null and status.',
                      'brier': 'Mean sum over all 13 ranks of squared normalized-probability minus one-hot error; not divided by 13.',
                      'repeats': 'An unordered mismatching position pair counts repeated only if it mismatched earlier in the same game.',
                      'choice_verification': 'All saved C probability overrides and every second-card max pending-rank choice with inclusive 1e-12 lexicographic ties are independently reconstructed. First-card choice is outside this diagnostic.'},
                  'limits': ['Exploratory metric choices after outcomes; no architecture or controller gate changes.',
                             'Query counts repeat positions and share 64 decks across fits; not independent statistical samples.',
                             'Raw argmax correctness does not imply the pair-scoring decision is correct. Unseen raw values are ignored by the actual picker.',
                             'Mean confidence minus accuracy is an aggregate diagnostic, not a complete calibration test or causal explanation.',
                             'No model forward, native replay, fitting, new seed or RNG call.'],
                  'new_model_calls': 0, 'new_native_calls': 0, 'new_rng_calls': 0,
                  'wall_seconds': time.monotonic() - begin}
        write(out / 'results.json', result)
        lines = ['# Saved C-policy confidence and choice diagnostic', '',
                 'Exploratory analysis of all 576 completed games across three families and all three fitted seeds. Existing gates are unchanged.', '',
                 '| Family | Second-card decisions | Mismatches | Unseen selected | Seen-hidden wrong top rank | Seen-hidden correct top rank, wrong pending rank | Visible selected |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for mode, row in families.items():
            counts = [row['mismatch_categories'][key]['mismatches'] for key in CATEGORIES]
            lines.append(f"| {mode} | {row['second_card_decisions']} | {row['actual_mismatches']} | "
                         + ' | '.join(str(n) for n in counts) + ' |')
        lines += ['', '| Family | Eligible hidden queries | Top-rank accuracy | Raw mean max confidence | NLL (nats) | Brier (13-rank sum) |',
                  '|---|---:|---:|---:|---:|---:|']
        for mode, row in families.items():
            q = row['all_eligible_seen_hidden_queries']
            nll = 'infinite' if q['mean_nll_nats'] is None else f"{q['mean_nll_nats']:.6f}"
            lines.append(f"| {mode} | {q['correct']}/{q['eligible_queries']} | {q['accuracy']:.4%} | "
                         f"{q['mean_raw_max_confidence']:.4%} | {nll} | {q['mean_brier_sum']:.6f} |")
        lines += ['', '| Family / mismatch category | Count | Repeats | Mean raw true-rank p | Mean raw pending-rank p | Mean raw max p |',
                  '|---|---:|---:|---:|---:|---:|']
        for mode, row in families.items():
            for key, stats in row['mismatch_categories'].items():
                if stats['mismatches']:
                    p = stats['probabilities']
                    lines.append(f"| {mode} / {key} | {stats['mismatches']} | {stats['repeat_mismatches']} | "
                                 + ' | '.join(f"{p[field]['mean']:.6f}" for field in FIELDS[:3]) + ' |')
        lines += ['', 'Raw unseen predictions are not used by C: the picker substitutes uniform 1/13 probabilities. Seen-hidden recall targets come only from earlier public frames. NLL and Brier use fixed row normalization, with no evaluation-set tuning. All per-fit denominators, repeated-mismatch probability summaries, age >32 queries and authenticated episode hashes are retained in results.json.', '',
                  'These are conditional saved-history diagnostics. A small number of recall errors can alter subsequent discovery and repeated choices; probability overlap can matter even when the top rank is correct. No counterfactual policy or architecture benefit was evaluated.', '',
                  f'Analysis source SHA-256: `{source_hash}`.', f'Execution completion SHA-256: `{COMPLETED}`.',
                  f'Results JSON SHA-256: `{sha(out / "results.json")}`.', '']
        with (out / 'short.md').open('x') as handle:
            handle.write('\n'.join(lines))
        write(out / 'completed.json', {'status': 'complete', 'source_sha256': source_hash,
                                      'execution_completed_sha256': COMPLETED, 'wall_seconds': time.monotonic() - begin,
                                      'files': {name: {'sha256': sha(out / name), 'bytes': (out / name).stat().st_size}
                                                for name in ('started.json', 'results.json', 'short.md')}})
    except BaseException as error:
        try:
            write(out / 'failed.json', {'status': 'failed', 'error': repr(error), 'source_sha256': source_hash,
                                       'wall_seconds': time.monotonic() - begin})
        except BaseException as secondary:  # noqa: BLE001 - retain original failure if receipt writing fails.
            if callable(getattr(error, 'add_note', None)):
                error.add_note(f'Failure receipt also failed: {secondary!r}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(**vars(parser.parse_args()))
