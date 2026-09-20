"""Explain saved typed-decision losses and errors without changing any decision."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import report_dialogue_typed_v2 as report

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'output/dialogue-typed-v1/training-01'
RUN_PIN = '65bd57084dbc7f7b99b782e140bae2e18f25b9e237e89a0132cead5fbfe2a095'
ANALYSIS = ROOT/'output/dialogue-typed-v1/analysis-02/summary.json'
ANALYSIS_PIN = '3969bcb7803d3f625a90a9e165a775bb9108f8301b9f225d198eed2e9e84a81c'
REPORTER_PIN = '6c97b4839352083885a3df5240cb1d3564a9652ec90ed30948a5bbada8b54365'
PAIRS = {'typing_balanced': ('typed_balanced', 'flat_balanced'),
         'typing_stratum': ('typed_stratum', 'flat_stratum'),
         'weighting_flat': ('flat_balanced', 'flat_stratum'),
         'weighting_typed': ('typed_balanced', 'typed_stratum')}
COMPONENTS = ('total', 'branch', 'value')
DECISIONS = ('correct', 'wrong_selected_branch', 'wrong_value', 'mass_branch_disagrees',
             'mass_branch_correct', 'conditional_value_correct', 'top1_tie')
require = report.require


def row_quantities(rows, packet):
    """Raw branch log mass and conditional value mass; never re-normalize."""
    _, choice, mass_error = report.predictions(rows, packet)
    logs = packet['log_probs'].astype(np.float64)
    branch_ids = np.full(logs.shape, -1, dtype=np.int64)
    for i, row in enumerate(rows):
        branch_ids[i, :row['candidate_count']] = np.minimum(row['candidate_types'], 2)
    branch_logs = np.empty((len(rows), 3), dtype=np.float64)
    for branch in range(3):
        supported = np.where(branch_ids == branch, logs, -np.inf)
        maximum = supported.max(axis=1)
        require(np.isfinite(maximum).all(), 'Each public branch has support')
        branch_logs[:, branch] = maximum+np.log(np.exp(supported-maximum[:, None]).sum(axis=1))
    index = np.arange(len(rows))
    target = np.asarray([r['current_label_index'] for r in rows])
    target_branch = branch_ids[index, target]
    selected_branch = branch_ids[index, choice]
    target_branch_log = branch_logs[index, target_branch]
    total = -logs[index, target]
    branch_loss = -target_branch_log
    value = target_branch_log-logs[index, target]
    residual = total-branch_loss-value
    require(np.isfinite(branch_logs).all() and np.isfinite(value).all(), 'Finite decomposed logs')
    require(np.abs(residual).max() <= 1e-10 and value.min() >= -1e-10, 'Exact loss decomposition')
    same_branch = branch_ids == target_branch[:, None]
    conditional_choice = np.where(same_branch, logs, -np.inf).argmax(axis=1)
    correct = choice == target
    wrong_branch = selected_branch != target_branch
    wrong_value = ~correct & ~wrong_branch
    require(np.all(correct.astype(int)+wrong_branch+wrong_value == 1), 'Exclusive actual error partition')
    require(np.all(target_branch[wrong_value] == 2), 'Singleton branch cannot have within-branch mistake')
    mass_branch = branch_logs.argmax(axis=1)
    return {'loss': {'total': total, 'branch': branch_loss, 'value': value},
            'decisions': {'correct': correct, 'wrong_selected_branch': wrong_branch, 'wrong_value': wrong_value,
                          'mass_branch_disagrees': mass_branch != selected_branch,
                          'mass_branch_correct': mass_branch == target_branch,
                          'conditional_value_correct': conditional_choice == target,
                          'top1_tie': (logs == logs.max(axis=1, keepdims=True)).sum(axis=1) > 1},
            'target_branch': target_branch, 'selected_branch': selected_branch,
            'error_code': np.where(correct, 0, np.where(wrong_branch, 1, 2)),
            'maximum_identity_error': float(np.abs(residual).max()),
            'maximum_probability_mass_error': mass_error,
            'minimum_branch_nll': float(branch_loss.min()), 'minimum_value_nll': float(value.min())}


def groups(rows):
    held = np.asarray([r['heldout_service'] for r in rows])
    changed = np.asarray([r['current_label_index'] != r['previous_current_index'] for r in rows])
    result = {panel+'/'+state: p & mask for panel, p in
              (('all', np.ones(len(rows), bool)), ('heldout_service', held), ('seen_service', ~held))
              for state, mask in (('all', np.ones(len(rows), bool)), ('changed', changed), ('retained', ~changed))}
    result.update({'primary/value/'+name: held & changed & np.asarray([r['current_value_group'] == name for r in rows])
                   for name in report.VALUES})
    result.update({'primary/service/'+name: held & changed & np.asarray([r['service'] == name for r in rows])
                   for name in sorted({r['service'] for r in rows if r['heldout_service']})})
    return result


def describe(rows, quantities, selected):
    grouping = report.layout(rows, selected)
    result = {k: grouping[k] for k in ('rows', 'services', 'dialogues', 'schema_queries')}
    result['loss'] = {c: report.means(quantities['loss'][c], grouping) for c in COMPONENTS}
    result['decisions'] = {key: int(quantities['decisions'][key][selected].sum()) for key in DECISIONS}
    flat = 3*quantities['target_branch'][selected]+quantities['selected_branch'][selected]
    result['selected_branch_confusion'] = np.bincount(flat, minlength=9).reshape(3, 3).tolist()
    require(sum(result['decisions'][k] for k in DECISIONS[:3]) == result['rows'], 'Group error partition')
    if result['rows']:
        for weighting in ('row', 'equal_service', 'equal_dialogue'):
            require(abs(result['loss']['total'][weighting]-result['loss']['branch'][weighting]
                        -result['loss']['value'][weighting]) < 1e-10, 'Pooled loss additivity')
    return result


def paired_group(rows, base, candidate, selected):
    grouping = report.layout(rows, selected)
    before, after = base['decisions']['correct'], candidate['decisions']['correct']
    masks = {'CC': before & after, 'CW': before & ~after, 'WC': ~before & after, 'WW': ~before & ~after}
    counts = {key: int((mask & selected).sum()) for key, mask in masks.items()}
    require(sum(counts.values()) == grouping['rows'], 'Paired correctness partition')
    net = counts['WC']-counts['CW']
    require(net == int(after[selected].sum())-int(before[selected].sum()), 'Paired net accuracy identity')
    transitions = np.bincount(3*base['error_code'][selected]+candidate['error_code'][selected], minlength=9).reshape(3, 3)
    return {'rows': grouping['rows'], 'correctness_counts': counts, 'error_transitions': transitions.tolist(),
            'loss_difference': {c: report.means(candidate['loss'][c]-base['loss'][c], grouping) for c in COMPONENTS},
            'accuracy_difference': net/grouping['rows'] if grouping['rows'] else None,
            'repair_rate': counts['WC']/(counts['WC']+counts['WW']) if counts['WC']+counts['WW'] else None,
            'harm_rate': counts['CW']/(counts['CW']+counts['CC']) if counts['CW']+counts['CC'] else None}


def contributions(rows, base, candidate, primary):
    """Subgroups share the FULL primary denominator, so contributions add up."""
    require(primary.any(), 'Nonempty primary panel')
    services = sorted({r['service'] for i, r in enumerate(rows) if primary[i]})
    weights = {'row': primary.astype(float)/primary.sum(), 'equal_service': np.zeros(len(rows))}
    for service in services:
        chosen = primary & np.asarray([r['service'] == service for r in rows])
        weights['equal_service'][chosen] = 1/(len(services)*chosen.sum())
    before, after = base['decisions']['correct'], candidate['decisions']['correct']
    partitions = {'by_correctness': {'CC': before & after, 'CW': before & ~after, 'WC': ~before & after, 'WW': ~before & ~after},
                  'by_target_type': {name: np.asarray([r['current_value_group'] == name for r in rows]) for name in report.VALUES}}
    output = {}
    for partition, masks in partitions.items():
        require(np.array_equal(sum((mask & primary).astype(int) for mask in masks.values()), primary.astype(int)), 'Contribution partition')
        output[partition] = {}
        for name, mask in masks.items():
            selected = mask & primary
            output[partition][name] = {'rows': int(selected.sum()), 'loss_difference': {
                c: {weight: float(np.dot(w[selected], (candidate['loss'][c]-base['loss'][c])[selected]))
                    for weight, w in weights.items()} for c in COMPONENTS}}
        for c in COMPONENTS:
            for weight, w in weights.items():
                total = math.fsum(cell['loss_difference'][c][weight] for cell in output[partition].values())
                expected = float(np.dot(w, candidate['loss'][c]-base['loss'][c]))
                require(abs(total-expected) <= 1e-10, 'Global denominator contribution identity')
    return output


def aggregate(rows, packets, published):
    require(set(packets) == report.FITS, 'Every final fit required')
    masks = groups(rows)
    quantities = {name: row_quantities(rows, packet) for name, packet in sorted(packets.items())}
    fits = {}
    for name, q in quantities.items():
        fits[name] = {'groups': {key: describe(rows, q, mask) for key, mask in masks.items()},
                      **{k: q[k] for k in ('maximum_identity_error', 'maximum_probability_mass_error',
                                           'minimum_branch_nll', 'minimum_value_nll')}}
        for panel in ('all', 'heldout_service', 'seen_service'):
            for state in ('all', 'changed', 'retained'):
                key = panel+'/'+state
                cell = fits[name]['groups'][key]
                old = published['fits'][name]['cells'][key]
                require(cell['rows'] == old['rows'], 'Published group rows')
                for weight in ('row', 'equal_service', 'equal_dialogue'):
                    a, b = cell['loss']['total'][weight], old['nll'][weight]
                    require(a == b or a is not None and b is not None and abs(a-b) < 1e-10, 'Published raw NLL')
                if cell['rows']:
                    require(abs(cell['decisions']['correct']/cell['rows']-old['accuracy']['row']) < 1e-12, 'Published actual accuracy')
    pairs = {}
    for label, (candidate, base) in PAIRS.items():
        pairs[label] = {}
        for seed in report.SEEDS:
            b, c = (quantities[f'{method}-{seed}'] for method in (base, candidate))
            pairs[label][str(seed)] = {'base': f'{base}-{seed}', 'candidate': f'{candidate}-{seed}',
                'groups': {key: paired_group(rows, b, c, mask) for key, mask in masks.items()},
                'primary_contributions': contributions(rows, b, c, masks['heldout_service/changed'])}
    return {'fits': fits, 'pairs': pairs, 'groups': list(masks), 'original_continuation_allowed': published['continuation_allowed'],
            'original_checks_passed': published['continuation']['checks_passed'], 'original_checks_total': published['continuation']['total_checks'],
            'scope': 'Posthoc saved-output algebra. No new model, probabilities, choices, gate, fitting or confirmation set. '
                     'Actual error type uses branch of selected candidate; branch-mass argmax is separately descriptive. '
                     'The correct branch supplied for conditional_value_correct is label-privileged, not a deployable controller. '
                     'Historical TRAIN exposure and privileged previous value remain. Separately trained models prevent causal representation claims.'}


def report_text(summary):
    lines = ['# Saved branch/value loss decomposition', '', summary['scope'], '',
             f"Original experiment remains FAIL ({summary['original_checks_passed']}/{summary['original_checks_total']} checks passed).", '',
             '| Fit | Total NLL | Branch NLL | Value NLL | Correct | Wrong branch | Wrong value |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, fit in summary['fits'].items():
        cell = fit['groups']['heldout_service/changed']
        losses = [f"{cell['loss'][c]['equal_service']:.6f}" for c in COMPONENTS]
        counts = [str(cell['decisions'][c]) for c in DECISIONS[:3]]
        lines.append('| '+name+' | '+' | '.join(losses+counts)+' |')
    lines += ['', ('Losses use equal-service weighting; counts cover the same 578 primary changed rows. '
              'Every fit and all descriptive contrasts are retained.'), '',
              '| Contrast | Seed | Branch loss change | Value loss change | Correct to wrong | Wrong to correct |',
              '|---|---:|---:|---:|---:|---:|']
    for label, seeds in summary['pairs'].items():
        for seed, pair in seeds.items():
            cell = pair['groups']['heldout_service/changed']; change = cell['loss_difference']; counts = cell['correctness_counts']
            lines.append(f"| {label} | {seed} | {change['branch']['equal_service']:.6f} | {change['value']['equal_service']:.6f} | {counts['CW']} | {counts['WC']} |")
    lines += ['', ('Wrong branch means the branch of the actual candidate argmax. Summed branch mass can prefer a different branch. '
              'No saved decision is changed. All per-service groups, target types, paired error transitions and additive contributions are in summary.json.')]
    return '\n'.join(lines)+'\n'


def execute(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    source_files = [Path(__file__), ROOT/'tests/test_diagnose_dialogue_typed.py',
                    ROOT/'research/dialogue-typed-decomposition-protocol.md', Path(report.__file__)]
    pins = {}
    try:
        pins = {str(p.relative_to(ROOT)): report.digest(p) for p in source_files}
        report.write(out/'started.json', {'run_sha256': RUN_PIN, 'published_summary_sha256': ANALYSIS_PIN,
                                       'source_sha256': pins, 'model_calls': 0})
        require(report.digest(Path(report.__file__)) == REPORTER_PIN, 'Bound authentication/aggregation helper')
        require(report.digest(ANALYSIS) == ANALYSIS_PIN, 'Published scientific result unchanged')
        published = report.read(ANALYSIS)
        require(published['technical_validity_passed'] is True and published['continuation_allowed'] is False
                and published['execution_completed_sha256'] == RUN_PIN, 'Closed complete failed study required')
        rows, done, _, _, size = report.authenticate_run(RUN, RUN_PIN)
        packets = {}
        for name in sorted(report.FITS):
            with np.load(RUN/'fits'/name/'predictions.npz', allow_pickle=False) as archive:
                packets[name] = {k: archive[k] for k in archive.files}
        summary = aggregate(rows, packets, published)
        summary.update(status='completed', version='dialogue-typed-decomposition-v1',
                       execution_completed_sha256=RUN_PIN, published_summary_sha256=ANALYSIS_PIN)
        report.check_manifest(RUN, done['files'], set(done['files']))
        require(report.digest(RUN/'completed.json') == RUN_PIN and report.digest(ANALYSIS) == ANALYSIS_PIN, 'End input stability')
        require(all(report.digest(ROOT/p) == pin for p, pin in pins.items()), 'End source stability')
        report.write(out/'summary.json', summary)
        with (out/'report.md').open('x') as stream: stream.write(report_text(summary))
        report.write(out/'receipt.json', {'status': 'completed', 'version': summary['version'],
            'execution_completed_sha256': RUN_PIN, 'published_summary_sha256': ANALYSIS_PIN,
            'source_sha256': pins, 'run_files': len(done['files'])+1, 'run_bytes': size,
            'files': {p.name: {'sha256': report.digest(p), 'bytes': p.stat().st_size} for p in out.iterdir() if p.is_file()},
            'original_continuation_allowed': False, 'model_calls': 0, 'training_calls': 0,
            'wall_seconds': time.monotonic()-start, 'scope': summary['scope']})
        return {'status': 'completed', 'original_continuation_allowed': False, 'summary_sha256': report.digest(out/'summary.json')}
    except BaseException as error:
        try:
            report.write(out/'failed.json', {'status': 'failed', 'error': repr(error), 'wall_seconds': time.monotonic()-start,
                                           'source_sha256': pins, 'model_calls': 0, 'training_calls': 0})
        except BaseException as save_error:  # noqa: BLE001 - preserve the original failure
            error.add_note('Could not preserve failure receipt: '+repr(save_error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    print(json.dumps(execute(parser.parse_args())))
