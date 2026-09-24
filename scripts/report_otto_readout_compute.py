"""Publish a closed compute comparison from saved JSON; never decode model data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = 'scripts/report_otto_readout_ablation.py'
HELPER_PIN = '23454ef229777e44533dcff4d9b9e42bdaa3ad649f0d900b443356f14e4eac81'
RUNNER = 'scripts/run_otto_readout_compute.py'
RUNNER_PIN = '47048d43fb76802213c33d42916745f6635fc8ff87bbd0b9a19d8afe48593ebb'
ARMS = ('pretrained', 'action_residual_only', 'full_joint')
LABELS = ('Pretrained parent', 'Residual: 74 epochs', 'Full joint: 40 epochs')
SEEDS = (309000001, 309000002, 309000003)
REGIMES = ('lambda3', 'lambda4')
ROLES = ('plan', 'producer_receipt', 'producer_terminal', 'audit_receipt', 'audit_terminal', 'audit')
ARCHIVE = 'https://github.com/kw2828/OpenJev/releases/tag/otto-readout-compute-v1'


def load(name, pin, alias):
    path = ROOT / name
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)) or hashlib.sha256(path.read_bytes()).hexdigest() != pin:
        raise ValueError('unchanged pinned publication helper: ' + name)
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def authenticate(h, path, sha):
    h.require(path.is_absolute() and h.descriptor(path)['sha256'] == sha, 'external closure pin')
    closure = h.read(path)
    h.require(closure['version'] == 'otto-readout-compute-closure-v1' and closure['technical_complete'] is True
              and closure['status'] in ('COMPARABLE_DEV_PASS', 'DEV_FAIL')
              and all(closure[k] is True for k in ('fresh_dev', 'held_out_from_training', 'development_only', 'prior_studies_remain_closed'))
              and all(closure[k] is False for k in ('dev_reused', 'test_admitted', 'confirmation_admitted', 'held_out_evidence'))
              and closure['closure_numerical_array_decodes'] == 0, 'closed fresh development comparison')
    records = {role: h.pinned(closure[role]) for role in ROLES}
    h.require(h.descriptor(closure['closure_source']['path']) == closure['closure_source'], 'original closure source pin')
    run = load(RUNNER, RUNNER_PIN, '_compute_report_runner')
    h.require(records['plan']['sources'][RUNNER] == RUNNER_PIN, 'registered runner source')
    for prefix, phase in (('producer', 'train'), ('audit', 'audit')):
        plan, receipt, directory = run.process_closure(closure['plan']['path'], closure[prefix + '_receipt']['path'], closure[prefix + '_terminal']['path'])
        h.require(plan == records['plan'] and receipt == records[prefix + '_receipt'] and receipt['phase'] == phase, 'same original phase')
        if phase == 'audit':
            h.require(receipt['files']['audit.json'] == closure['audit'] and Path(closure['audit']['path']) == directory / 'audit.json', 'closed audit payload')
    h.require(records['audit_receipt']['producer_receipt'] == closure['producer_receipt']
              and records['audit_receipt']['producer_terminal'] == closure['producer_terminal']
              and records['producer_terminal']['finished_ns'] <= records['audit_terminal']['started_ns'], 'original producer/audit join')
    audit = records['audit']
    counts = {'array_decodes': 22, 'checkpoint_decodes': 9, 'views_completed': 9, 'fits_checked': 6,
              'model_calls': 0, 'optimizer_calls': 0, 'teacher_calls': 0, 'native_calls': 0,
              'test_array_decodes': 0, 'journal_updates_checked': 3078, 'episode_exposures_checked': 18468}
    h.require(audit['version'] == 'otto-readout-compute-saved-audit-v1' and audit['agreement'] is True
              and audit['counts'] == records['audit_receipt']['audit_counts'] == closure['counts'] == counts
              and audit['status'] == 'completed_pending_original_audit_closure' and audit['technical_complete'] is False
              and audit['requires_original_supervisor_closure'] is True
              and audit['test_admitted'] is audit['confirmation_admitted'] is False, 'complete independently audited outputs')
    reports = {(r['family'], r['seed']): r for r in audit['reports']}
    h.require(len(audit['reports']) == len(reports) == 9 and set(reports) == {(a, s) for a in ARMS for s in SEEDS}, 'all nine views')
    manifest = next(iter(reports.values()))['identity_manifest']
    h.require(len(manifest) == 36 and all(type(row['length']) is int and row['length'] > 0 for row in manifest), 'all fresh paths')
    for report in reports.values():
        h.require(report['identity_manifest'] == manifest and report['stage'] == 'dev' and report['query_period'] == 4 and report['episodes'] == 36, 'matched fresh DEV views')
        for regime in REGIMES:
            for scope in ('full', 'initial', 'later'):
                leaf = report['scopes'][scope]['by_regime'][regime]
                h.require(leaf['episodes'] == 18 and leaf['declared_case_count'] == 6, 'complete case denominators')
                for metric in ('raw_gap', 'agreement', 'centered_mse'):
                    h.finite(leaf['case_weighted_' + metric])
            h.finite(report['prequery']['by_regime'][regime]['case_weighted_centered_mse'])
    result = audit['comparisons']
    h.require(result == closure['diagnostic'] and result['candidate'] == 'full_joint' and result['margin'] == .05
              and len(result['cells']) == result['total_cells'] == 6
              and {(c['seed'], c['regime']) for c in result['cells']} == {(s, r) for s in SEEDS for r in REGIMES}, 'all saved paired cells')
    for cell in result['cells']:
        for scope in ('full', 'later'):
            h.require(cell['gaps'][scope] == {arm: h.gap(reports, arm, cell['seed'], cell['regime'], scope) for arm in ARMS}, 'saved gap agreement')
        for key in ('later_gain_at_least_5pct', 'full_gap_nonregression'):
            h.require(set(cell[key]) == set(ARMS[:2]) and all(type(v) is bool for v in cell[key].values()), 'all four saved checks')
        h.require(cell['passed'] is all(cell[k][a] for a in ARMS[:2] for k in ('later_gain_at_least_5pct', 'full_gap_nonregression')), 'saved cell outcome')
    costs = {}
    for key, arms in (('measured_fit_seconds', ARMS[1:]), ('measured_view_seconds', ARMS)):
        costs[key] = {(r['family'], r['seed']): h.finite(r['seconds']) for r in audit[key]}
        h.require(len(audit[key]) == len(costs[key]) == len(arms) * 3 and set(costs[key]) == {(a, s) for a in arms for s in SEEDS}, 'complete measured costs')
    timing = result['compute_comparability']
    h.require(timing['bounds'] == [.9, 1.1] and len(timing['pairs']) == 3 and {p['seed'] for p in timing['pairs']} == set(SEEDS), 'all timing ratios')
    for pair in timing['pairs']:
        h.require(pair['residual_seconds'] == costs['measured_fit_seconds']['action_residual_only', pair['seed']]
                  and pair['full_joint_seconds'] == costs['measured_fit_seconds']['full_joint', pair['seed']]
                  and pair['full_joint_seconds'] > 0 and h.finite(pair['ratio']) == pair['residual_seconds'] / pair['full_joint_seconds']
                  and pair['passed'] is (.9 <= pair['ratio'] <= 1.1), 'saved timing identity')
    h.require(result['efficacy_passed'] is all(c['passed'] for c in result['cells'])
              and result['passed_cells'] == sum(c['passed'] for c in result['cells'])
              and timing['passed'] is all(p['passed'] for p in timing['pairs'])
              and result['overall_passed'] is (result['efficacy_passed'] and timing['passed'])
              and closure['status'] == ('COMPARABLE_DEV_PASS' if result['overall_passed'] else 'DEV_FAIL'), 'preserved efficacy and cost decision')
    collection_record = records['plan']['bridge_inputs']['collection_receipt']
    collection = h.pinned(collection_record)
    for receipt in (records['producer_receipt'], records['audit_receipt'], collection):
        h.finite(receipt['wall_seconds'])
    h.require(not any(k in sys.modules for k in ('numpy', 'torch', 'tensorflow', 'jax', 'mlx')), 'JSON-only authentication')
    return closure, records, reports, costs, collection


def render(path, sha, output):
    h = load(HELPER, HELPER_PIN, '_compute_report_metadata_helpers')
    h.require(output.is_absolute() and output.is_relative_to(ROOT) and '..' not in output.parts
              and output.parent.is_dir() and not output.exists() and not any(p.is_symlink() for p in (output, *output.parents)), 'fresh contained report directory')
    closure, records, reports, costs, collection = authenticate(h, path, sha)
    own = h.descriptor(__file__)
    output.mkdir()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    result = records['audit']['comparisons']
    cells = sorted(result['cells'], key=lambda c: (c['seed'], c['regime']))
    pairs = sorted(result['compute_comparability']['pairs'], key=lambda p: p['seed'])
    fig = plt.figure(figsize=(15, 13), layout='constrained')
    grid = fig.add_gridspec(3, 2, height_ratios=(1, 1, 1.15))
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
    colors, markers = ('#1764a0', '#bf6613', '#358342'), ('o', 's', '^')
    def dots(ax, arms, labels, value):
        for i, seed in enumerate(SEEDS):
            ax.scatter([value(a, seed) for a in arms], [y + (i - 1) * .14 for y in range(len(arms))], c=colors[i], marker=markers[i], label=str(seed), zorder=3)
        ax.scatter([h.mean([value(a, s) for s in SEEDS]) for a in arms], range(len(arms)), c='#222222', marker='D', label='Equal fit mean', zorder=4)
        ax.set_yticks(range(len(arms)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, max(value(a, s) for a in arms for s in SEEDS) * 1.10)
        ax.grid(axis='x', color='#dddddd')
        ax.spines[['right', 'top']].set_visible(False)
    for ax, regime in zip(axes[:2], REGIMES, strict=True):
        dots(ax, ARMS, LABELS, lambda a, s, regime=regime: h.gap(reports, a, s, regime, 'later'))
        ax.set(title=regime + ': later nonquery teacher-score gap', xlabel='Case-weighted raw gap (lower is better)')
    axes[0].legend(fontsize=8, title='Fit seed; shared 36 fresh DEV paths')
    dots(axes[2], ARMS[1:], LABELS[1:], lambda a, s: costs['measured_fit_seconds'][a, s])
    axes[2].set(title='Measured continuation training time', xlabel='Seconds per completed fit (including recorded harness work)')
    axes[3].axvspan(.9, 1.1, color='#d6e9d8', label='Registered comparable interval')
    for i, pair in enumerate(pairs):
        axes[3].scatter(pair['ratio'], i, c=colors[i], marker=markers[i], zorder=3)
        axes[3].annotate(f" {pair['ratio']:.4f}: {'PASS' if pair['passed'] else 'FAIL'}", (pair['ratio'], i), xytext=(6, 5), textcoords='offset points', fontsize=9)
    axes[3].set(yticks=range(3), yticklabels=[str(p['seed']) for p in pairs], title='Actual cost comparability, separately tested', xlabel='Residual74 / full-joint40 elapsed fit time')
    axes[3].set_xlim(min(.8, min(p['ratio'] for p in pairs) - .12), max(1.4, max(p['ratio'] for p in pairs) + .3))
    axes[3].set_ylim(-.25, 2.35)
    axes[3].legend(fontsize=8, loc='lower right')
    ax = fig.add_subplot(grid[2, :])
    flags = [[int(c[k][a]) for a in ARMS[:2] for k in ('later_gain_at_least_5pct', 'full_gap_nonregression')] for c in cells]
    ax.imshow(flags, cmap=ListedColormap(('#f5d6d6', '#d6e9d8')), vmin=0, vmax=1, aspect='auto')
    for i, row in enumerate(flags):
        for j, value in enumerate(row):
            ax.text(j, i, 'PASS' if value else 'FAIL', ha='center', va='center')
    ax.set(xticks=range(4), xticklabels=['vs parent: later >=5% gain', 'vs parent: full nonregression', 'vs residual74: later >=5% gain', 'vs residual74: full nonregression'],
           yticks=range(6), yticklabels=[f"{c['seed']} / {c['regime']}" for c in cells], title='All 24 efficacy checks; no failed cell excluded')
    fig.suptitle(f"OTTO compute comparison: {closure['status']}\nEfficacy: {result['passed_cells']}/6 cells; cost comparability: {sum(p['passed'] for p in pairs)}/3 seeds", fontsize=16, weight='bold')
    fig.savefig(output / 'methods.png', dpi=160)
    plt.close(fig)
    def outcome(passed):
        return 'PASS' if passed else 'FAIL'

    def metric(a, s, r, scope):
        return h.fmt(h.gap(reports, a, s, r, scope))
    text = ['# OTTO readout versus recurrent adaptation at calibrated cost', '',
            f"**{closure['status']}. Efficacy: {result['passed_cells']}/6 cells. Actual cost comparability: {sum(p['passed'] for p in pairs)}/3 seeds.**", '',
            ('The 116-parameter residual head trained for 74 epochs (666 updates); full joint trained 6,112 parameters for 40 epochs (360 updates). Both start from the same three parents and 54 TRAIN paths. Epoch counts were fixed before fresh collection, without live time-based stopping or intermediate checkpoint selection.'), '',
            (f"All nine views use the same 36 fresh DEV paths ({sum(r['length'] for r in next(iter(reports.values()))['identity_manifest']):,} rows), six cases per setting and three collectors. These paths were held out from training. This remains fixed-path development evidence, not autonomous performance, independent confirmation, statistical equivalence or an architecture novelty claim."), '',
            '![Every fit, score, measured cost ratio, and efficacy check](methods.png)', '',
            'Dots show all three fit seeds and diamonds their equal means. They share evaluation paths; no confidence interval or independent-data claim is inferred from these three fits.', '',
            '## All nine views', '', h.table(['Arm', 'Fit seed', 'lambda3 full', 'lambda3 later', 'lambda4 full', 'lambda4 later', 'View seconds'],
                [[a, s, *(metric(a, s, r, scope) for r in REGIMES for scope in ('full', 'later')), h.fmt(costs['measured_view_seconds'][a, s])] for a in ARMS for s in SEEDS]), '',
            'Lower case-weighted raw gap is better. Full includes every nonquery action; later includes nonquery steps >=5. Complete support and supplemental metrics remain in the audited JSON.', '',
            '## Every paired comparison', '', h.table(['Seed', 'Setting', 'Supported later cases', 'Parent later/full', 'Residual later/full', 'Cell'],
                [[c['seed'], c['regime'], f"{c['supported_later_cases']}/6", *(' / '.join(outcome(c[k][a]) for k in ('later_gain_at_least_5pct', 'full_gap_nonregression')) for a in ARMS[:2]), outcome(c['passed'])] for c in cells]), '',
            'Each cell requires a >=5% later-gap reduction with strict improvement and no full-gap regression against both controls. A zero comparator gap cannot pass strict improvement. Overall continuation also requires all three actual time ratios within [0.90, 1.10].', '',
            '## All six fit times and three cost ratios', '', h.table(['Seed', 'Residual74 seconds', 'Full-joint40 seconds', 'Residual / joint', 'Comparable'],
                [[p['seed'], h.fmt(p['residual_seconds']), h.fmt(p['full_joint_seconds']), h.fmt(p['ratio']), outcome(p['passed'])] for p in pairs]), '',
            (f"Saved worker wall times: fresh collection {h.fmt(collection['wall_seconds'])} s; producer {h.fmt(records['producer_receipt']['wall_seconds'])} s; independent audit {h.fmt(records['audit_receipt']['wall_seconds'])} s. Fit timing includes construction, training, validation and checkpoint serialization in its recorded interval. Shared original pretraining and collection are excluded from fit ratios. These are measurements of the qualified CPU implementations, not optimal implementations or inference-speed benchmarks."), '',
            '## Evidence', '', 'Original closure: ' + h.link(path, output) + ' (`' + sha + '`).', '',
            *['- ' + role.replace('_', ' ') + ': ' + (f'[audit.json in the complete evidence archive]({ARCHIVE})' if role == 'audit' else h.link(closure[role]['path'], output)) + ' (`' + closure[role]['sha256'] + '`)' for role in ROLES], '',
            'Both original process closures were authenticated. Publication reads saved JSON and opaque payload hashes, with no array/checkpoint decoding, model calls, optimizer updates or teacher/simulator calls. Prior studies remain closed; TEST and confirmation are unadmitted.', '']
    (output / 'README.md').write_text('\n'.join(text))
    for role in ROLES:
        h.require(h.descriptor(closure[role]['path']) == closure[role], 'publication input unchanged')
    collection_record = records['plan']['bridge_inputs']['collection_receipt']
    h.require(h.descriptor(collection_record['path']) == collection_record
              and h.descriptor(HELPER)['sha256'] == HELPER_PIN and h.descriptor(RUNNER)['sha256'] == RUNNER_PIN
              and h.descriptor(path)['sha256'] == sha and h.descriptor(__file__) == own, 'unchanged inputs and publication sources')
    receipt = {'version': 'otto-readout-compute-report-v1', 'status': 'completed', 'scientific_status': closure['status'],
               'closure': h.descriptor(path), 'inputs': {r: closure[r] for r in ROLES}, 'renderer': own, 'helper': h.descriptor(HELPER),
               'collection_receipt': collection_record,
               'views_displayed': 9, 'fit_costs_displayed': 6, 'efficacy_checks_displayed': 24, 'timing_ratios_displayed': 3,
               'publication_array_decodes': 0, 'publication_checkpoint_decodes': 0, 'publication_model_calls': 0,
               'publication_training_updates': 0, 'publication_teacher_calls': 0, 'publication_simulator_calls': 0,
               'files': {name: h.descriptor(output / name) for name in ('README.md', 'methods.png')}}
    with (output / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return h.descriptor(output / 'receipt.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closure', type=Path, required=True)
    parser.add_argument('--closure-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.closure, args.closure_sha256, args.output), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
