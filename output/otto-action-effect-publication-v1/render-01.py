"""Render authenticated closed action-effect JSON; no scientific execution.

Prepared prospectively. Do not execute until root supplies the completed closure
SHA. Plotting imports occur after original metadata/JSON closure authentication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'output/otto-action-effect-v1'
DEST = ROOT / 'research/otto-action-effect-results'
DOCUMENT = ROOT / 'research/otto-action-effect-results.md'
FAMILIES = ('effect_recurrent', 'paired_recurrent', 'paired_blind', 'paired_direct')
LABELS = ('Effect recurrent', 'Paired recurrent', 'Paired action blind', 'Paired direct')
SEEDS = (330000001, 330000002, 330000003)
REGIMES = ('lambda3', 'lambda4')
FIELDS = ('effect_error', 'sampled_nll', 'decision_gap', 'normal_nll', 'normal_decision_gap')
COLORS = ('#1765af', '#d7802f', '#59844b', '#8d659b')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def descriptor(path):
    require(path.is_file() and not path.is_symlink(), 'regular saved evidence')
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def consume(directory, receipt, name):
    path = directory / name
    require(descriptor(path) == receipt['files'][name], 'unchanged closed JSON: ' + name)
    return read(path)


def value(number, *, optional=False):
    if number is None and optional:
        return None
    require(type(number) in (int, float) and math.isfinite(number) and number >= 0, 'finite saved metric')
    return float(number)


def average(values):
    require(bool(values), 'nonempty paired values')
    return None if any(v is None for v in values) else math.fsum(values) / len(values)


def relative(candidate, control):
    return None if candidate is None or control is None or control == 0 else 100 * (control - candidate) / control


def num(v):
    return 'undefined' if v is None else f'{v:.6f}'


def percent(v):
    return 'undefined' if v is None else f'{v:+.2f}%'


def compact_gate(gate):
    return {key: gate[key] for key in ('version', 'candidate', 'controls', 'fit_seeds', 'regimes', 'thresholds',
                                      'cells', 'passed_cells', 'total_cells', 'passed')}


def authenticate(closure_sha):
    require(descriptor(STUDY / 'closure-01.json')['sha256'] == closure_sha, 'externally supplied scientific closure')
    closure = read(STUDY / 'closure-01.json')
    require(closure['version'] == 'otto-action-effect-v1' and closure['technical_complete'] is True
            and closure['independent_audit_passed'] is True and closure['status'] in ('DEV_PASS', 'DEV_FAIL')
            and set(closure['processes']) == {'collection', 'fit', 'audit'}, 'all completed original phases')
    receipts, terminals = {}, {}
    for phase in ('collection', 'fit', 'audit'):
        directory = STUDY / f'{phase}-01'
        terminal_path = STUDY / f'{phase}-native-01.terminal.json'
        require(descriptor(directory / 'receipt.json') == closure['processes'][phase]['receipt']
                and descriptor(terminal_path) == closure['processes'][phase]['terminal'], 'original closure byte joins')
        receipt, terminal = read(directory / 'receipt.json'), read(terminal_path)
        started = consume(directory, receipt, 'started.json')
        launch = started['launch']
        require(receipt['status'] == terminal['status'] == 'completed' and terminal['returncode'] == 0
                and terminal['group_absent'] is True and not terminal['timed_out']
                and terminal['error'] is None and terminal['clock_error'] is None
                and terminal['timing_available'] is True and terminal['cleanup']['reaped'] is True
                and terminal['cleanup']['group_absent'] is True and terminal['cleanup']['errors'] == []
                and all(terminal[key] == item for key, item in launch.items())
                and launch['started_ns'] <= receipt['started_ns'] < receipt['finished_ns'] <= terminal['finished_ns'] <= terminal['deadline_ns']
                and terminal['deadline_ns'] == terminal['started_ns'] + terminal['cap_seconds'] * 10**9
                and receipt['plan_sha256'] == started['plan_sha256'], 'genuine bounded process completion')
        launch_path = Path(terminal['command'][terminal['command'].index('--supervision') + 1])
        require(descriptor(launch_path) == receipt['supervision'], 'original launch evidence')
        receipts[phase], terminals[phase] = receipt, terminal
    require(len({r['plan_sha256'] for r in receipts.values()}) == 1
            and terminals['collection']['finished_ns'] <= terminals['fit']['started_ns']
            and terminals['fit']['finished_ns'] <= terminals['audit']['started_ns'], 'same registration and ordered phase closures')
    audit = consume(STUDY / 'audit-01', receipts['audit'], 'audit.json')
    fit = consume(STUDY / 'fit-01', receipts['fit'], 'summary.json')
    collection = consume(STUDY / 'collection-01', receipts['collection'], 'summary.json')
    require(audit['agreement'] is True and audit['requires_original_supervisor_closure'] is True
            and audit['gate'] == closure['gate'] and fit['status'] == closure['status'], 'same independently audited result')
    require(audit['inputs']['collection_receipt'] == closure['processes']['collection']['receipt']
            and audit['inputs']['fit_receipt'] == closure['processes']['fit']['receipt'], 'audit producer identities')
    require(audit['training_reference'] == fit['training_reference'] == collection['training_reference'], 'same charged upstream TRAIN')
    for name in ('gate', 'legacy_gate'):
        gate = audit[name]
        require(gate['candidate'] == FAMILIES[0] and gate['controls'] == list(FAMILIES[1:])
                and gate['fit_seeds'] == list(SEEDS) and gate['regimes'] == list(REGIMES)
                and len(gate['cells']) == gate['total_cells'] == 18
                and gate['passed_cells'] == sum(c['passed'] for c in gate['cells'])
                and gate['passed'] is all(c['passed'] for c in gate['cells']), 'complete separately reported gate')
        if name in closure:
            require(closure[name] == gate, 'same closed ' + name)
    require(closure['status'] == ('DEV_PASS' if audit['gate']['passed'] else 'DEV_FAIL'), 'new gate defines registered outcome')
    reports = {(r['family'], r['fit_seed'], r['condition']): r for r in audit['reports']}
    effects = {(r['family'], r['fit_seed'], r['regime']): r for r in audit['gate']['long_effects']}
    require(len(reports) == len(audit['reports']) == 24
            and set(reports) == {(f, s, p) for f in FAMILIES for s in SEEDS for p in ('gap', 'normal')}
            and len(effects) == len(audit['gate']['long_effects']) == 24
            and set(effects) == {(f, s, r) for f in FAMILIES for s in SEEDS for r in REGIMES}, 'all saved fit views')
    return closure, audit, fit, collection, reports, effects


def extract(closure, audit, fit, collection, reports, effects):
    rows = []
    for family in FAMILIES:
        for seed in SEEDS:
            for regime in REGIMES:
                gap = reports[family, seed, 'gap']['groups']['long']['by_regime'][regime]
                normal = reports[family, seed, 'normal']['groups']['all']['by_regime'][regime]
                effect = effects[family, seed, regime]
                rows.append({'family': family, 'fit_seed': seed, 'regime': regime,
                    'effect_error': value(effect['case_weighted_model_effect_error']),
                    'oracle_signal': value(effect['case_weighted_oracle_signal']),
                    'sampled_nll': value(gap['case_weighted_log_score']),
                    'decision_gap': value(gap['case_weighted_decision_gap'], optional=True),
                    'normal_nll': value(normal['case_weighted_log_score']),
                    'normal_decision_gap': value(normal['case_weighted_decision_gap'], optional=True),
                    'declared_cases': gap['declared_cases'], 'decision_supported_cases': gap['supported_cases'],
                    'decision_rows': gap['decision_rows'], 'outcome_rows': gap['outcome_rows']})
    means = [{'family': f, 'regime': r,
              **{key: average([x[key] for x in rows if x['family'] == f and x['regime'] == r])
                 for key in (*FIELDS, 'oracle_signal')}} for f in FAMILIES for r in REGIMES]
    by = {(r['family'], r['fit_seed'], r['regime']): r for r in rows}
    pairs = [{'control': f, 'fit_seed': s, 'regime': r,
              **{k: relative(by[FAMILIES[0], s, r][k], by[f, s, r][k]) for k in FIELDS}}
             for f in FAMILIES[1:] for r in REGIMES for s in SEEDS]
    upstream = fit['training_reference']['teacher_cost_linkage']
    worker_times = {name: value(row['worker_seconds']) for name, row in closure['processes'].items()}
    return {'gate': compact_gate(audit['gate']), 'legacy_gate': compact_gate(audit['legacy_gate']),
        'per_seed': rows, 'means': means, 'paired_relative_reductions_pct': pairs,
        'costs': {'new_worker_seconds': worker_times, 'new_worker_total_seconds': math.fsum(worker_times.values()),
                 'upstream_parent_acquisition': upstream,
                 'new_plus_upstream_acquisition_worker_seconds': math.fsum(worker_times.values()) + value(upstream['whole_parent_collection_seconds']),
                 'fits': [{k: r[k] for k in ('family', 'seed', 'seconds', 'updates', 'exposures', 'effect_weight')} for r in fit['fits']],
                 'prediction_times': fit['prediction_times']},
        'training_reference': fit['training_reference'], 'fresh_dev_counts': collection['counts']['dev'],
        'prefix_exclusions': collection['prefix_exclusions'], 'collection_calls': collection['calls'],
        'aggregation': 'Equal fit-seed means on shared originating cases; positive paired reductions favor effect_recurrent.',
        'case_selection': False, 'effect_case_filtering': False}


def render_figure(numbers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.7), layout='constrained')
    measures = (('effect_error', 'Signed action-effect error E'), ('sampled_nll', 'Sampled log loss'),
                ('decision_gap', 'Teacher-action cost gap'))
    for row, regime in enumerate(REGIMES):
        for column, (metric, label) in enumerate(measures):
            ax = axes[row, column]
            for index, (family, color) in enumerate(zip(FAMILIES, COLORS, strict=True)):
                values = [r[metric] for r in numbers['per_seed'] if r['family'] == family and r['regime'] == regime]
                mean = average(values)
                if mean is None:
                    ax.text(index, .02, 'undefined', rotation=90, ha='center', transform=ax.get_xaxis_transform())
                    continue
                ax.bar(index, mean, color=color, alpha=.82, width=.68)
                ax.annotate(f'{mean:.4f}', (index, max(values)), xytext=(0, 6),
                            textcoords='offset points', ha='center', va='bottom', fontsize=8)
                for i, item in enumerate(values):
                    ax.scatter(index + .13 * (i - 1), item, marker=('o', '^', 's')[i], s=27,
                               facecolor='white', edgecolor='#253746', linewidth=.8, zorder=3)
            if metric == 'effect_error':
                signal = next(r['oracle_signal'] for r in numbers['means'] if r['regime'] == regime)
                ax.axhline(signal, color='#253746', ls=':', lw=1.2)
                ax.text(.02, .97, 'Dotted: oracle action signal S', va='top', fontsize=8, transform=ax.transAxes)
            ax.set_title(f'{regime} | {label}', loc='left', fontweight='bold')
            ax.set_xticks(range(4), ('Effect\nrecurrent', 'Paired\nrecurrent', 'Paired\nblind', 'Paired\ndirect'))
            ax.set_ylim(bottom=0)
            ax.margins(y=.16)
            ax.grid(axis='y', color='#dbe2e9', lw=.7)
            ax.set_axisbelow(True)
    new, old = numbers['gate'], numbers['legacy_gate']
    new_status, old_status = ('PASS' if g['passed'] else 'FAIL' for g in (new, old))
    fig.suptitle(f"Paired action-effect supervision | new gate {new_status} {new['passed_cells']}/18; legacy gate {old_status} {old['passed_cells']}/18\n"
                 'Fresh DEV, horizons 5-8 | lower is better | bars: three-fit means; points: individual fits',
                 fontsize=15, fontweight='bold')
    fig.savefig(DEST / 'benchmark.png', dpi=160)
    plt.close(fig)


def document(numbers):
    gate, legacy = numbers['gate'], numbers['legacy_gate']
    outcome = 'DEV_PASS' if gate['passed'] else 'DEV_FAIL'
    labels = dict(zip(FAMILIES, LABELS, strict=True))
    same = [r for r in numbers['paired_relative_reductions_pct'] if r['control'] == 'paired_recurrent']
    e_wins = sum(r['effect_error'] is not None and r['effect_error'] > 0 for r in same)
    e_gate = sum(r['effect_error'] is not None and r['effect_error'] >= 10 for r in same)
    gap_gate = sum(r['decision_gap'] is not None and r['decision_gap'] >= 5 for r in same)
    lines = ['# Paired action-effect supervision: ' + outcome, '',
             f"**New mechanism gate: {gate['passed_cells']}/18. Unchanged legacy forecast gate: {legacy['passed_cells']}/18.** "
             'All 12 fits and the independent saved-output audit completed. The new gate determines this registration\'s '
             'outcome; the legacy gate is reported separately and cannot rescue it.', '',
             f"Against the paired recurrent control, the candidate lowers long-horizon effect error in **{e_wins}/6** "
             f"seed/regime pairs, reaches the registered 10% reduction in **{e_gate}/6**, and reaches the 5% "
             f"decision-gap reduction in **{gap_gate}/6**. The complete comparisons below retain every seed.", '',
             '![All four methods, two settings and three fits](otto-action-effect-results/benchmark.png)', '',
             'The candidate and paired recurrent control use the same 1,046 TRAIN blocks, initializations, epoch orders, '
             'three forward rollouts per batch, probability targets and ordinary losses. Only the candidate gives weight '
             '0.1 to the TRAIN-normalized squared signed-effect error. This comparison tests that extra loss; it does not '
             'isolate a new architecture. All controls already receive both factual and opposite-action soft targets.', '',
             'Bars are means of three fits on the same fresh DEV cases. Points are individual fits, not independent '
             'datasets or confidence intervals. E compares the direction and magnitude of the probability change '
             'under action XOR1. Every case remains, including cases with zero oracle signal.', '',
             '## Long-horizon means', '',
             '| Method | Setting | Effect error E | Sampled NLL | Decision gap | Normal NLL | Normal decision gap |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in numbers['means']:
        lines.append('| ' + ' | '.join([labels[row['family']], row['regime'], *[num(row[k]) for k in FIELDS]]) + ' |')
    lines.extend(['', 'E, sampled negative log likelihood and decision gap above cover horizons 5-8. Normal scores '
                  'cover horizons 1-8. Outcome scores include all absorbing suffixes; decision scores average nonterminal '
                  'rows within case and then cases with decision support. Undefined values are never filled with zero.', '',
                  'The new gate requires at least 10% lower E and 5% lower decision gap, no more than 1% higher long '
                  'NLL or normal NLL/gap, and at least 48 supported cases, against all controls in all seeds/settings. '
                  'The unchanged legacy gate instead requires 1% lower long NLL and 5% lower decision gap, with its '
                  'same normal limits. These are different hypotheses, not interchangeable pass counts.', '',
                  '## Every paired comparison', '',
                  'Positive percentages favor effect recurrent: `100*(control-candidate)/control`. Each row is a '
                  'paired seed comparison, not a ratio of aggregated means. A zero denominator is undefined.', '',
                  '| Seed | Setting | Control | E reduction | NLL reduction | Gap reduction | Normal NLL reduction | Normal gap reduction |',
                  '|---|---|---|---:|---:|---:|---:|---:|'])
    for row in numbers['paired_relative_reductions_pct']:
        lines.append('| ' + ' | '.join([str(row['fit_seed']), row['regime'], labels[row['control']],
                                      *[percent(row[k]) for k in FIELDS]]) + ' |')
    lines.extend(['', '## Gate detail', '', '| Seed | Setting | Control | New gate | Legacy gate | Failed new conditions |',
                  '|---|---|---|---|---|---|'])
    old_cells = {(c['fit_seed'], c['regime'], c['control']): c for c in legacy['cells']}
    for cell in gate['cells']:
        failed = ', '.join(c['name'] for c in cell['conditions'] if not c['passed']) or 'none'
        old = old_cells[cell['fit_seed'], cell['regime'], cell['control']]
        lines.append(f"| {cell['fit_seed']} | {cell['regime']} | {cell['control']} | {cell['passed']} | {old['passed']} | {failed} |")
    counts = numbers['fresh_dev_counts']
    lines.extend(['', '## Cases and charged computation', '',
                  f"Fresh DEV retained **{counts['lambda3']}/128 lambda3** and **{counts['lambda4']}/128 lambda4** "
                  f"originating cases. The {numbers['prefix_exclusions']} prefix terminations were excluded without "
                  'replacement. Results condition on surviving the fixed eight-transition analytic prefix. '
                  'TRAIN is the unchanged 1,046-case parent dataset; no earlier DEV, TEST or confirmation arrays were reused.', ''])
    for regime in REGIMES:
        r = next(r for r in numbers['per_seed'] if r['family'] == FAMILIES[0] and r['regime'] == regime)
        lines.append(f"* {regime}: {r['declared_cases']} cases for effects/outcomes; {r['decision_supported_cases']} with "
                     f"decision support; {r['decision_rows']}/{r['outcome_rows']} nonterminal long-horizon rows.")
    lines.extend(['', '| Method | Mean fit seconds | Fit range |', '|---|---:|---:|'])
    for family in FAMILIES:
        times = [value(r['seconds']) for r in numbers['costs']['fits'] if r['family'] == family]
        require(len(times) == 3, 'all recorded fit timings')
        lines.append(f"| {labels[family]} | {average(times):.3f} | {min(times):.3f}-{max(times):.3f} |")
    costs = numbers['costs']
    upstream = costs['upstream_parent_acquisition']
    lines.extend(['', 'New worker times: ' + '; '.join(f'{k} {v:.3f}s' for k, v in costs['new_worker_seconds'].items()) + '.',
                  f"Total new worker time is **{costs['new_worker_total_seconds']:.3f}s**. Adding the charged upstream "
                  f"parent collection (**{upstream['whole_parent_collection_seconds']:.3f}s**, "
                  f"{upstream['whole_parent_native_steps']} steps and {upstream['whole_parent_teacher_annotations']} teacher "
                  f"annotations) gives **{costs['new_plus_upstream_acquisition_worker_seconds']:.3f}s** of accounted worker "
                  'time. The upstream charge includes that parent collection\'s unused DEV work; labels are not free. '
                  'Qualification, publication and historical pretraining are outside this worker-time total.', '',
                  'Recorded times include this implementation\'s checks and serialization. Nested annotation/Bayesian call '
                  'timings must not be added to wall times. There is no hardware-independent speed claim.', '',
                  '## Interpretation boundaries', '',
                  'This is a forecasting and teacher-action imitation study along precommitted paths. It is not autonomous '
                  'control, RL, connectome evidence, robotics transfer or architectural novelty. Full-belief targets are '
                  'privileged training annotations, never neural inference inputs. Normal terminal certainty uses only '
                  'already observed found; blind forecasts receive no future observation.', '',
                  'The independent audit reconstructs saved DEV probabilities, effects, scalar scores and both gates. '
                  'It does not replay optimization or native scalar primitives, and TRAIN normalization is authenticated '
                  'metadata rather than an independent TRAIN-array recomputation. Three fits share the same evaluation '
                  'cases. Gate counts cannot be compared with prior studies as a progress metric.', '',
                  '[Compact scalars and individual fits](otto-action-effect-results/summary.json) | '
                  '[Protocol](otto-action-effect-protocol.md) | '
                  '[Original closure](../output/otto-action-effect-v1/closure-01.json)', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closure-sha256', required=True)
    args = parser.parse_args()
    closure, audit, fit, collection, reports, effects = authenticate(args.closure_sha256)
    require(not DEST.exists() and not DOCUMENT.exists(), 'exclusive publication outputs')
    numbers = extract(closure, audit, fit, collection, reports, effects)
    text = document(numbers)
    DEST.mkdir()
    render_figure(numbers)
    with DOCUMENT.open('x') as stream:
        stream.write(text)
    with (DEST / 'summary.json').open('x') as stream:
        json.dump(numbers, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    receipt = {'scope': 'authenticated closed JSON publication only', 'closure': descriptor(STUDY / 'closure-01.json'),
        'audit': descriptor(STUDY / 'audit-01/audit.json'), 'renderer': descriptor(Path(__file__)),
        'fit_summary': descriptor(STUDY / 'fit-01/summary.json'), 'collection_summary': descriptor(STUDY / 'collection-01/summary.json'),
        'files': {str(p.relative_to(ROOT)): descriptor(p) for p in [DOCUMENT, *DEST.iterdir()]},
        'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0, 'native_calls': 0, 'teacher_calls': 0,
        'optimizer_calls': 0, 'case_selection': False}
    with (DEST / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps({'new_gate': [audit['gate']['passed_cells'], 18], 'legacy_gate': [audit['legacy_gate']['passed_cells'], 18],
                      'report': str(DOCUMENT), 'figure': str(DEST / 'benchmark.png')}))


if __name__ == '__main__':
    main()
