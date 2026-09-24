"""Publish only closed scalar diagnostics and opaque preserved evidence."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'output/finite-decision-error-v1'
PLAN = FOLDER / 'registration.json'
PLAN_SHA = 'ff1008ba9b36b265e40dc8101a3f3417a6f1348b34d5e4a57d5bcf0095352721'
OUTPUT = ROOT / 'research/finite-decision-error-results'
OVERVIEW = ROOT / 'research/finite-decision-error-results.md'
PARENT = ROOT / 'research/finite-head-learning-results'
VERSION = 'finite-decision-error-publication-v1'
ARMS = ('rounded_anchor', 'rounded_random', 'matched_free_random')
CONTROLS = ('matched_free_random', 'rounded_anchor')
SEEDS = tuple(range(436261001, 436261006))
HORIZONS = (1, 2, 4, 8)
CATEGORIES = ('candidate_better', 'control_better', 'equal')
BIN_LABELS = ('[0,0.001)', '[0.001,0.01)', '[0.01,0.1)', '[0.1,inf)')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path.absolute(),
            'canonical ordinary input file')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def compare(actual, expected):
    """Compare saved scalar outputs only, without recomputing any case metric."""
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), 'same saved object fields')
        for key in expected:
            compare(actual[key], expected[key])
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), 'same saved list length')
        for left, right in zip(actual, expected, strict=True):
            compare(left, right)
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
                'saved scalar agreement within frozen audit tolerance')
    else:
        require(type(actual) is type(expected) and actual == expected, 'same saved identity')


def authenticate():
    require(re.fullmatch('[0-9a-f]{64}', PLAN_SHA), 'publisher remains unbound until registration')
    require(descriptor(PLAN)['sha256'] == PLAN_SHA, 'exact original child registration')
    plan = read(PLAN)
    require(len(plan['sources']) == 144 and all(descriptor(ROOT / name) == pin
            for name, pin in plan['sources'].items()), 'all registered sources before helper imports')
    from run_finite_decision_error import closed, files, validate
    validate(plan, PLAN_SHA)
    closures = {phase: closed(plan, PLAN_SHA, phase) for phase in ('qualify', 'diagnose', 'audit')}
    receipts, terminals = {}, {}
    for phase in closures:
        spec = plan['phases'][phase]
        receipts[phase] = read(spec['output'] + '.receipt.json')
        terminals[phase] = read(spec['supervision'].replace('.launch.json', '.terminal.json'))
        expected = {'npz_opens': 0 if phase == 'qualify' else 2,
                    'array_decodes': 0 if phase == 'qualify' else 17,
                    'model_calls': 0, 'optimizer_calls': 0, 'generator_calls': 0}
        require(receipts[phase]['counts'] == expected, 'exact original phase input accounting')
    for first, second in (('qualify', 'diagnose'), ('diagnose', 'audit')):
        require(terminals[first]['clock_backend'] == terminals[second]['clock_backend']
                and terminals[first]['finished_ns'] <= terminals[second]['started_ns'],
                'same-clock original phase ordering')
    require(receipts['diagnose']['qualify_closure'] == closures['qualify']
            and receipts['audit']['qualify_closure'] == closures['qualify']
            and receipts['audit']['diagnose_closure'] == closures['diagnose'],
            'each numerical phase binds its actual predecessor closure')
    require(receipts['diagnose']['result'] == {'case_records': 29160, 'cases': 486}
            and receipts['audit']['result'] == {'raw_vector_joins': 29160, 'agreement': True},
            'complete original records and source-vector joins')
    child_files = files(FOLDER)
    # Current diagnostic metrics are first read after all original closures.
    diagnostic = read(Path(plan['phases']['diagnose']['output']) / 'summary.json')
    audit = read(Path(plan['phases']['audit']['output']) / 'audit.json')
    require(diagnostic['version'] == 'finite-decision-error-diagnostic-v1'
            and diagnostic['descriptive_only'] is True and diagnostic['failed_rule_rescued'] is False
            and audit['version'] == 'finite-decision-error-audit-v1' and audit['agreement'] is True
            and audit['technical_complete'] is False and audit['requires_original_supervisor_closure'] is True
            and audit['descriptive_only'] is True and audit['failed_rule_rescued'] is False
            and audit['original_case_ids_bound'] is True
            and audit['all_population_contributions_reconciled'] is True,
            'independently audited descriptive-only record')
    roster = diagnostic['roster']
    require(roster['arms'] == list(ARMS) and set(roster['controls']) == set(CONTROLS)
            and roster['candidate'] == 'rounded_random' and roster['seeds'] == list(SEEDS)
            and roster['horizons'] == list(HORIZONS) and roster['long_horizons'] == [4, 8]
            and roster['cases'] == len(roster['case_ids']) == len(set(roster['case_ids'])) == 486,
            'all original arms, seeds, horizons and retained cases')
    require(audit['counts'] == {'case_records_recomputed': 29160, 'model_horizon_means': 60,
            'parent_mean_joins': 120, 'paired_case_contrasts': 9720, 'contrasts': 20,
            'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0, 'optimizer_calls': 0,
            'world_or_generator_calls': 0}, 'complete pure scalar-audit scope')
    for field in ('means', 'arm_means', 'contrast_means'):
        compare(diagnostic[field], audit[field])
    compare([{key: value for key, value in row.items() if key != 'records'}
             for row in diagnostic['contrasts']], audit['contrasts'])
    require(len(audit['means']) == 60 and {(r['arm'], r['seed'], r['horizon']) for r in audit['means']}
            == {(arm, seed, h) for arm in ARMS for seed in SEEDS for h in HORIZONS}
            and len(audit['arm_means']) == 12 and len(audit['contrast_means']) == 4,
            'all sixty policy-horizon groups and descriptive means')
    require(len(audit['contrasts']) == 20 and {(r['control'], r['seed'], r['horizon']) for r in audit['contrasts']}
            == {(control, seed, h) for control in CONTROLS for seed in SEEDS for h in (4, 8)},
            'all twenty paired groups without selection')
    for row in (*audit['contrasts'], *audit['contrast_means']):
        require([r['category'] for r in row['categories']] == list(CATEGORIES)
                and [r['label'] for r in row['margin_bins']] == list(BIN_LABELS),
                'all categories and fixed margin bins including empty bins')
    parent_receipt = read(PARENT / 'receipt.json')
    parent_summary = read(PARENT / 'summary.json')
    require(descriptor(PARENT / 'receipt.json') == plan['parent']['receipt']
            and descriptor(PARENT / 'summary.json') == plan['parent']['summary']
            and parent_summary['audit']['advance']['passed'] is False
            and len(parent_summary['audit']['advance']['conditions']) == 15
            and sum(parent_summary['audit']['advance']['conditions'].values()) == 14,
            'original parent fourteen-of-fifteen failure unchanged')
    external = {'receipt': {'path': str(PARENT / 'receipt.json'), **plan['parent']['receipt']},
                'archive': {'path': str(PARENT / 'evidence.tar.gz'), **parent_receipt['files']['evidence.tar.gz']},
                'parent_status': parent_summary['status'], 'passing_conditions': 14, 'conditions': 15,
                'included_in_child_archive': False}
    log = (Path(plan['phases']['qualify']['output']) / 'command-1.log').read_text()
    completions = re.findall(r'(?m)^\s*(\d+) passed(?:, (\d+) warnings?)? in [^\n]+$', log)
    require(len(completions) == 1, 'one complete original selected-test result')
    return {'plan': plan, 'closures': closures, 'phase_receipts': receipts,
            'phase_seconds': {phase: terminal['wall_seconds'] for phase, terminal in terminals.items()},
            'diagnostic': diagnostic, 'audit': audit, 'child_files': child_files,
            'external_parent': external,
            'qualification': {'tests_passed': int(completions[0][0]),
                              'warnings': int(completions[0][1] or 0)}}


def chart(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), layout='constrained')
    indexed = {(r['control'], r['seed'], r['horizon']): r for r in summary['contrasts']}
    means = {(r['control'], r['horizon']): r for r in summary['contrast_means']}
    labels = {'matched_free_random': 'Matched free random (primary control)',
              'rounded_anchor': 'Rounded anchor (descriptive control)'}
    for column, horizon in enumerate((4, 8)):
        for row_index, control in enumerate(CONTROLS):
            axis = axes[row_index, column]
            rows = [indexed[control, seed, horizon] for seed in SEEDS] + [means[control, horizon]]
            negative = [next(r['population_contributions']['delta_regret'] for r in x['categories']
                             if r['category'] == 'candidate_better') for x in rows]
            positive = [next(r['population_contributions']['delta_regret'] for r in x['categories']
                             if r['category'] == 'control_better') for x in rows]
            delta = [x['means']['delta_regret'] for x in rows]
            xs = list(range(6))
            axis.bar(xs, negative, color='#257a65', width=.68, label='Candidate-better contribution')
            axis.bar(xs, positive, color='#c17430', width=.68, label='Control-better contribution')
            axis.scatter(xs[:5], delta[:5], color='#192635', s=22, zorder=4, label='Net difference')
            axis.scatter([5], [delta[5]], color='#192635', marker='D', s=36, zorder=4, label='Five-policy mean')
            tick_labels = [f'{seed}\nΔ={value:+.3e}' for seed, value in zip(SEEDS, delta[:5], strict=True)]
            axis.set_xticks(xs, [*tick_labels, f'Mean\nΔ={delta[5]:+.3e}'], fontsize=8)
            axis.set_yscale('linear')
            axis.axhline(0, color='#667080', linewidth=.8)
            axis.set_title(f'H{horizon}: {labels[control]}', fontsize=11)
            axis.set_ylabel('Candidate minus control regret\n(full-population contribution)', fontsize=9)
            axis.grid(axis='y', color='#e5e9ee', alpha=.8)
            axis.set_axisbelow(True)
            axis.spines[['top', 'right']].set_visible(False)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='outside lower center', ncol=2, frameon=False, fontsize=9)
    fig.suptitle('Every paired policy: opposing case contributions, with numeric net differences\n'
                 'Retrospective only; parent continuation remains FAIL (14/15)', fontsize=13, weight='bold')
    fig.savefig(path, dpi=180, metadata={'Description':
        'All twenty paired groups and four means. Every panel uses a linear axis; tick labels retain each '
        'net difference even when bars are visually small. Negative favors rounded random. '
        'Shared-case policy means are descriptive, not independent replications. Saved audited JSON only.'})
    plt.close(fig)


def fmt(value):
    return 'none (empty)' if value is None else f'{value:.9g}'


def document(summary):
    failed = next(r for r in summary['contrasts']
                  if r['control'] == 'matched_free_random' and r['seed'] == 436261004 and r['horizon'] == 8)
    group = {(r['arm'], r['seed'], r['horizon']): r['means'] for r in summary['means']}
    first, second = group['rounded_random', 436261004, 8], group['matched_free_random', 436261004, 8]
    lines = ['# Where prediction error and decision regret diverge', '',
        '**Retrospective diagnostic complete. Parent continuation remains FAIL, 14/15 conditions.**', '',
        ('This reuses the closed BASE predictions for all three models, all five fit seeds and all 486 retained '
         'cases. No model is trained, selected or run again. Every H1/H2/H4/H8 policy group and both '
         'H4/H8 control comparisons remain in the report and evidence.'), '',
        '![All paired seeds and signed case contributions](benchmark.png)', '',
        ('The paired difference is rounded-random regret minus control regret. Green bars sum cases where '
         'the candidate has lower regret; orange bars sum cases where the control has lower regret. '
         'Both use the full 486-case denominator. Their sum is the net mean difference, shown numerically '
         'for every seed. Exact-equal cases contribute zero and remain in all denominators. Each panel uses '
         'a linear axis; panel scales can differ. Diamonds describe the same five policies on shared cases, '
         'not a new ensemble or five independent populations.'), '',
        'The preidentified failed comparison:', '',
        (f"Seed 436261004 at H8 has candidate regret {fmt(first['regret'])} versus {fmt(second['regret'])}, "
         f"a difference of {fmt(failed['means']['delta_regret'])}. Its four-action MSE is "
         f"{fmt(first['mse'])} versus {fmt(second['mse'])}; centered MSE is "
         f"{fmt(first['centered_mse'])} versus {fmt(second['centered_mse'])}. "
         'This comparison was named before the diagnostic decoded its inputs because it failed the parent '
         'rule. It receives no separate threshold, fitting, replacement or eligibility decision.'), '',
        '| Case category | Cases | Actions differ | Full-population regret-difference contribution | Within-category mean difference |',
        '|---|---:|---:|---:|---:|']
    for row in failed['categories']:
        within = None if row['within_means'] is None else row['within_means']['delta_regret']
        lines.append(f"| {row['category']} | {row['count']} | {row['actions_differ']} | {fmt(row['population_contributions']['delta_regret'])} | {fmt(within)} |")
    high_margin = failed['margin_bins'][-1]
    other_bins = math.fsum(row['population_contributions']['delta_regret'] for row in failed['margin_bins'][:-1])
    lines += ['', (f"The two policies choose different actions on {failed['actions_differ']}/{failed['cases']} cases. "
        f"In the predeclared true-margin >=0.1 bin, {high_margin['actions_differ']} of "
        f"{high_margin['count']} cases have different actions; that bin contributes "
        f"{high_margin['population_contributions']['delta_regret']:+.12g} to the full-population regret difference. "
        f"The other three bins together contribute {other_bins:+.12g}. "
        'The reversal is therefore not confined to tiny near-ties. True cost columns are centered by the '
        'world definition, and learned columns use 0.25 minus four-action softmax probabilities. Both cost '
        'vectors are therefore centered by construction. Their raw and action-centered MSE agree at the '
        'saved precision, as expected; centering is not a new correction or an independent improvement. '
        'These findings use the fixed bins and '
        'all cases; they do not change the failed continuation rule.'), '',
        ('Lower average squared error need not preserve the ordering of the lowest costs in every '
        'case. The decomposition shows associations in saved predictions; it does not identify which parameter '
        'or training mechanism caused them. The predicted chosen-versus-true-best contrast is a ranking '
        'witness, not confidence or a calibrated probability. No significance test, new loss selection or '
        'architecture advancement follows from this diagnostic.'), '',
        (f"The selected qualification passed {summary['qualification']['tests_passed']} tests with "
         f"{summary['qualification']['warnings']} warning(s). Original native phase times were "
         f"{summary['phase_seconds']['qualify']:.6f}s for qualification, "
         f"{summary['phase_seconds']['diagnose']:.6f}s for diagnosis and "
         f"{summary['phase_seconds']['audit']:.6f}s for audit. All original processes closed successfully."), '',
        ('The producer and audit each open two original NPZ files and decode 17 arrays: IDs, true costs and '
         '15 blind-cost predictions. Total new scientific input work is four NPZ opens and 34 array decodes. '
         'The audit independently joins all 29,160 original vectors and IDs, recomputes all case metrics and '
         '9,720 paired case contrasts, and reconciles 120 regret/MSE values with all 60 published parent rows. '
         'Its pure scalar subroutine performs no file decoding; those reads belong to its enclosing phase. '
         'No phase invokes a model, optimizer or world generator. Publication plots saved scalar JSON only.'), '',
        'All paired category decompositions:', '',
        '| Control | Seed | H | Category | Cases | Actions differ | Candidate contribution | Control contribution | Difference contribution | Within-category difference |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    ordered = sorted(summary['contrasts'], key=lambda r: (CONTROLS.index(r['control']), r['seed'], r['horizon']))
    for paired in ordered:
        for row in paired['categories']:
            values = row['population_contributions']
            within = None if row['within_means'] is None else row['within_means']['delta_regret']
            lines.append(f"| {paired['control']} | {paired['seed']} | {paired['horizon']} | {row['category']} | {row['count']} | {row['actions_differ']} | {fmt(values['candidate_regret'])} | {fmt(values['control_regret'])} | {fmt(values['delta_regret'])} | {fmt(within)} |")
    lines += ['', 'All fixed true-margin bins, including empty bins:', '',
        '| Control | Seed | H | True margin | Cases | Actions differ | Candidate contribution | Control contribution | Difference contribution | Within-bin difference |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for paired in ordered:
        for row in paired['margin_bins']:
            values = row['population_contributions']
            within = None if row['within_means'] is None else row['within_means']['delta_regret']
            lines.append(f"| {paired['control']} | {paired['seed']} | {paired['horizon']} | {row['label']} | {row['count']} | {row['actions_differ']} | {fmt(values['candidate_regret'])} | {fmt(values['control_regret'])} | {fmt(values['delta_regret'])} | {fmt(within)} |")
    lines += ['', ('Margin is the second-smallest true cost minus the smallest, retaining exact ties. '
        'The four intervals were fixed before decoding. A category contribution uses all 486 cases, '
        'whereas its within-group mean uses only its own count; they must not be interchanged.'), '',
        'All sixty policy-horizon means:', '',
        '| Model | Seed | H | Cases | Regret | Four-action MSE | Centered MSE | True margin | Predicted contrast |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in sorted(summary['means'], key=lambda r: (ARMS.index(r['arm']), r['seed'], r['horizon'])):
        values = row['means']
        lines.append(f"| {row['arm']} | {row['seed']} | {row['horizon']} | {row['cases']} | "
                     + ' | '.join(fmt(values[name]) for name in ('regret', 'mse', 'centered_mse', 'true_margin', 'predicted_contrast')) + ' |')
    lines += ['', ('[Protocol](../finite-decision-error-protocol.md) · [Saved summary](summary.json) · '
        '[Diagnostic evidence](evidence.tar.gz) · [Manifest](manifest.json) · [Publication receipt](receipt.json) · '
        '[Failed parent study](../finite-head-learning-results.md) · [Parent evidence](../finite-head-learning-results/evidence.tar.gz)'), '',
        ('The child archive contains the entire registered diagnostic, its complete 144-source snapshot, all three '
         'phase closures, case JSON, and this publication. The original parent archive is separately linked '
         'and pinned in the manifest; it is not duplicated. Installed runtime dependencies remain external. '
         'The parent original qualification failure and final scientific failure remain preserved in its evidence.'), '']
    return '\n'.join(lines)


def archive(auth):
    from run_finite_decision_error import publish
    paths = {'study/' + name: (FOLDER / name, pin) for name, pin in auth['child_files'].items()}
    for name in ('summary.json', 'report.md', 'benchmark.png'):
        paths['publication/' + name] = (OUTPUT / name, descriptor(OUTPUT / name))
    paths['publication/overview.md'] = (OVERVIEW, descriptor(OVERVIEW))
    paths['publication/publisher.py'] = (Path(__file__).resolve(), descriptor(Path(__file__).resolve()))
    manifest = {'version': VERSION, 'registration': {'path': str(PLAN), **descriptor(PLAN)},
        'files': {name: {'original_path': str(path), **pin} for name, (path, pin) in sorted(paths.items())},
        'external_parent': auth['external_parent'],
        'scope': 'Complete child study and publication; original parent archive and installed runtime are external.',
        'self_exclusion': 'Manifest is archived but excludes its own hash; final publication receipt is separate.'}
    publish(OUTPUT / 'manifest.json', manifest)
    paths['publication/manifest.json'] = (OUTPUT / 'manifest.json', descriptor(OUTPUT / 'manifest.json'))
    with ((OUTPUT / 'evidence.tar.gz').open('xb') as raw,
          gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped,
          tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as saved):
        for name, (path, pin) in sorted(paths.items()):
            require(descriptor(path) == pin, 'unchanged archive input')
            data = path.read_bytes()
            require({'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} == pin, 'opaque input pin')
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            saved.addfile(info, io.BytesIO(data))
    with tarfile.open(OUTPUT / 'evidence.tar.gz', 'r:gz') as saved:
        require(saved.getnames() == sorted(paths), 'complete exact deterministic archive roster')
        for item in saved.getmembers():
            path, pin = paths[item.name]
            require(item.isfile() and saved.extractfile(item).read() == path.read_bytes()
                    and descriptor(path) == pin, 'all opaque members roundtrip unchanged')
    return len(paths)


def main():
    require(not OUTPUT.exists() and not OVERVIEW.exists(), 'exclusive publication paths')
    source = descriptor(Path(__file__).resolve())
    auth = authenticate()
    from run_finite_decision_error import files, publish
    summary = {'version': VERSION, 'status': 'DESCRIPTIVE_ONLY_PARENT_FAIL', 'technical_complete': True,
        'registration': {'path': str(PLAN), **descriptor(PLAN)}, 'parent': auth['external_parent'],
        'roster': auth['diagnostic']['roster'], 'definitions': auth['diagnostic']['definitions'],
        **{key: auth['audit'][key] for key in ('means', 'arm_means', 'contrasts', 'contrast_means')},
        'audit': auth['audit'], 'phase_seconds': auth['phase_seconds'], 'qualification': auth['qualification'],
        'phase_counts': {phase: row['counts'] for phase, row in auth['phase_receipts'].items()},
        'counts_scope': 'Phase counts include source-file decodes; audit.counts describe its pure scalar subroutine only.',
        'model_selection': False, 'failed_rule_rescued': False, 'new_experiment_admission': False}
    OUTPUT.mkdir()
    publish(OUTPUT / 'summary.json', summary)
    chart(summary, OUTPUT / 'benchmark.png')
    report = document(summary)
    with (OUTPUT / 'report.md').open('x') as stream:
        stream.write(report)
    overview = report.replace('(benchmark.png)', '(finite-decision-error-results/benchmark.png)')
    for name in ('summary.json', 'evidence.tar.gz', 'manifest.json', 'receipt.json'):
        overview = overview.replace('(' + name + ')', '(finite-decision-error-results/' + name + ')')
    overview = overview.replace('(../finite-', '(finite-')
    with OVERVIEW.open('x') as stream:
        stream.write(overview)
    members = archive(auth)
    after = authenticate()
    require(after['child_files'] == auth['child_files'] and after['closures'] == auth['closures']
            and after['external_parent'] == auth['external_parent']
            and descriptor(Path(__file__).resolve()) == source, 'all input bytes and publisher unchanged afterward')
    publish(OUTPUT / 'receipt.json', {'version': VERSION, 'status': 'PASS', 'registration': summary['registration'],
        'publisher': source, 'files': files(OUTPUT), 'overview': {'path': str(OVERVIEW), **descriptor(OVERVIEW)},
        'archive_members': members, 'sources': auth['plan']['sources'], 'child_files': auth['child_files'],
        'phase_closures': auth['closures'], 'external_parent': auth['external_parent'],
        'publication_counts': dict.fromkeys(('npz_opens', 'array_decodes', 'checkpoint_decodes',
                                            'model_calls', 'optimizer_calls', 'generator_calls', 'audit_reruns'), 0),
        'plotting_scope': 'Matplotlib uses audited saved JSON scalars only after original phase closure.',
        'result_reads_after_original_closure': True, 'inputs_unchanged': True,
        'failed_rule_rescued': False, 'new_experiment_admission': False, 'visual_review_required': True})
    print(json.dumps({'output': str(OUTPUT), 'status': summary['status'], 'archive_members': members}))


if __name__ == '__main__':
    main()
