"""Render a closed direct-readout diagnostic using authenticated saved JSON only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = 'scripts/report_otto_readout_ablation.py'
HELPER_PIN = '23454ef229777e44533dcff4d9b9e42bdaa3ad649f0d900b443356f14e4eac81'
RUNNER = 'scripts/run_otto_direct_readout.py'
RUNNER_PIN = 'b2a1ec928623f575fef50fa53f3acfbc96a16c5cce01d94f0d95d7d1d34c1a1e'
SEEDS = (309000001, 309000002, 309000003)
ARMS = ('pretrained', 'action_residual_only', 'ols', 'ridge', 'full_joint')
LABELS = ('Pretrained parent', 'Adam residual: 74 epochs', 'OLS residual', 'Ridge residual', 'Full joint: 40 epochs')
REGIMES = ('lambda3', 'lambda4')
ROLES = ('plan', 'producer_receipt', 'producer_terminal', 'audit_receipt', 'audit_terminal', 'audit')


def load(name, pin, alias):
    path = ROOT / name
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)) or hashlib.sha256(path.read_bytes()).hexdigest() != pin:
        raise ValueError('unchanged pinned reporting source: ' + name)
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def authenticate(h, path, sha):
    h.require(path.is_absolute() and h.descriptor(path)['sha256'] == sha, 'external closure pin')
    closure = h.read(path)
    h.require(closure['version'] == 'otto-direct-readout-closure-v1' and closure['technical_complete'] is True
              and closure['status'] in ('REUSED_DEV_PASS', 'REUSED_DEV_FAIL')
              and all(closure[k] is True for k in ('dev_reused', 'held_out_from_training', 'prior_studies_remain_closed'))
              and all(closure[k] is False for k in ('fresh_dev', 'held_out_evidence', 'test_admitted', 'confirmation_admitted'))
              and closure['closure_numerical_array_decodes'] == 0, 'closed reused-development diagnostic')
    records = {role: h.pinned(closure[role]) for role in ROLES}
    h.require(h.descriptor(closure['closure_source']['path']) == closure['closure_source'], 'unchanged closure source')
    run = load(RUNNER, RUNNER_PIN, '_direct_report_runner')
    h.require(records['plan']['sources'][RUNNER] == RUNNER_PIN, 'registered runner identity')
    for prefix, phase in (('producer', 'train'), ('audit', 'audit')):
        plan, receipt, directory = run.process_closure(closure['plan']['path'], closure[prefix + '_receipt']['path'],
                                                       closure[prefix + '_terminal']['path'])
        h.require(plan == records['plan'] and receipt == records[prefix + '_receipt'] and receipt['phase'] == phase,
                  'same original successful process')
        if phase == 'audit':
            h.require(receipt['files']['audit.json'] == closure['audit']
                      and Path(closure['audit']['path']) == directory / 'audit.json', 'closed saved audit payload')
    h.require(records['audit_receipt']['producer_receipt'] == closure['producer_receipt']
              and records['audit_receipt']['producer_terminal'] == closure['producer_terminal']
              and records['producer_terminal']['finished_ns'] <= records['audit_terminal']['started_ns'],
              'original producer and later audit join')
    audit = records['audit']
    counts = {'array_decodes': 49, 'checkpoint_decodes': 15, 'cache_decodes': 3, 'views_completed': 27,
              'train_views_completed': 12, 'dev_views_completed': 15, 'fits_checked': 6,
              'model_calls': 0, 'optimizer_calls': 0, 'solver_calls': 0, 'teacher_calls': 0,
              'native_calls': 0, 'test_array_decodes': 0}
    h.require(audit['version'] == 'otto-direct-readout-saved-audit-v1' and audit['agreement'] is True
              and audit['status'] == 'completed_pending_original_audit_closure' and audit['technical_complete'] is False
              and audit['requires_original_supervisor_closure'] is True
              and audit['counts'] == records['audit_receipt']['audit_counts'] == closure['counts'] == counts,
              'complete independently audited outputs')
    reports = {(r['stage'], r['family'], r['seed']): r for r in audit['reports']}
    roster = {(stage, arm, seed) for stage, arms in (('train', ARMS[:4]), ('dev', ARMS)) for arm in arms for seed in SEEDS}
    h.require(len(audit['reports']) == len(reports) == 27 and set(reports) == roster, 'all 27 reports retained')
    for stage, n in (('train', 54), ('dev', 36)):
        manifest = reports[stage, 'pretrained', SEEDS[0]]['identity_manifest']
        h.require(len(manifest) == n, 'complete evaluation paths')
        for (split, _, _), report in reports.items():
            if split != stage:
                continue
            h.require(report['query_period'] == 4 and report['episodes'] == n
                      and report['identity_manifest'] == manifest, 'matched chronological reports')
            h.finite(report['scopes']['full']['overall']['episode_weighted_centered_mse'])
            h.finite(report['prequery']['overall']['episode_weighted_centered_mse'])
            for regime in REGIMES:
                for scope in ('full', 'later'):
                    h.finite(report['scopes'][scope]['by_regime'][regime]['case_weighted_raw_gap'])
    fits_pin = records['producer_receipt']['files']['fits.json']
    fits_list = h.pinned(fits_pin)
    fits = {(r['family'], r['seed']): r for r in fits_list}
    checks = {(r['family'], r['seed']): r for r in audit['fit_checks']}
    expected_fits = {(a, s) for a in ('ols', 'ridge') for s in SEEDS}
    h.require(len(fits_list) == len(audit['fit_checks']) == len(fits) == len(checks) == 6
              and set(fits) == set(checks) == expected_fits, 'all six adaptation records')
    for key, row in fits.items():
        h.require(row['parity']['passed'] is True and math.isclose(row['exported_cached_loss'],
                  checks[key]['exported_cached_loss'], rel_tol=1e-8, abs_tol=1e-11),
                  'audited export parity and cached loss')
        for field in ('cache_design_seconds', 'solve_seconds', 'export_seconds', 'train_validation_seconds',
                      'standalone_adaptation_seconds'):
            h.finite(row[field])
        h.finite(row['cache_metadata']['seconds'])
        for field in ('parent_loss', 'adam74_loss', 'data_loss', 'regularization', 'exported_cached_loss'):
            h.finite(checks[key][field])
    decision = audit['comparisons']
    cells = decision['cells']
    h.require(decision == closure['diagnostic'] and decision['candidate'] == 'ridge'
              and decision['controls'] == list(ARMS[:2]) and decision['margin'] == .05
              and len(cells) == decision['total_cells'] == 6
              and {(c['seed'], c['regime']) for c in cells} == {(s, r) for s in SEEDS for r in REGIMES},
              'all original paired ridge comparisons')
    for cell in cells:
        for scope in ('full', 'later'):
            expected = {a: reports['dev', a, cell['seed']]['scopes'][scope]['by_regime'][cell['regime']]['case_weighted_raw_gap'] for a in ARMS}
            h.require(cell['gaps'][scope] == expected, 'paired gaps agree with saved reports')
        flags = [cell[k][a] for a in ARMS[:2] for k in ('later_gain_at_least_5pct', 'full_gap_nonregression')]
        h.require(all(type(v) is bool for v in flags) and cell['passed'] is all(flags), 'complete saved cell decision')
    passed = all(c['passed'] for c in cells)
    h.require(decision['passed_cells'] == sum(c['passed'] for c in cells) and decision['overall_passed'] is passed
              and closure['status'] == ('REUSED_DEV_PASS' if passed else 'REUSED_DEV_FAIL'), 'preserved diagnostic outcome')
    h.require(not any(n in sys.modules for n in ('numpy', 'torch', 'tensorflow', 'jax', 'mlx')), 'JSON-only authentication')
    return closure, records, reports, fits, checks, fits_pin


def render(path, sha, output):
    h = load(HELPER, HELPER_PIN, '_direct_report_helpers')
    h.require(output.is_absolute() and output.is_relative_to(ROOT) and '..' not in output.parts
              and output.parent.is_dir() and not output.exists()
              and not any(p.is_symlink() for p in (output, *output.parents)), 'exclusive contained report output')
    closure, records, reports, fits, checks, fits_pin = authenticate(h, path, sha)
    own = h.descriptor(__file__)
    # Numerical plotting libraries are imported only after metadata authentication.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    output.mkdir()
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), layout='constrained')
    colors, markers = ('#1764a0', '#bf6613', '#358342'), ('o', 's', '^')
    def dots(ax, keys, labels, value):
        for i, seed in enumerate(SEEDS):
            ax.scatter([value(key, seed) for key in keys], [j + (i - 1) * .14 for j in range(len(keys))],
                       c=colors[i], marker=markers[i], label=str(seed), zorder=3)
        ax.scatter([h.mean([value(k, s) for s in SEEDS]) for k in keys], range(len(keys)),
                   c='#222222', marker='D', label='Equal fit mean', zorder=4)
        ax.set_yticks(range(len(keys)), labels)
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.grid(axis='x', color='#dddddd')
        ax.spines[['right', 'top']].set_visible(False)

    def gap(arm, seed, regime, scope):
        return reports['dev', arm, seed]['scopes'][scope]['by_regime'][regime]['case_weighted_raw_gap']

    keys = [(a, scope) for a in ARMS for scope in ('full', 'later')]
    labels = [label + ' / ' + scope for label in LABELS for scope in ('full', 'later')]
    for ax, regime in zip(axes[0], REGIMES, strict=True):
        dots(ax, keys, labels, lambda key, seed, r=regime: gap(key[0], seed, r, key[1]))
        ax.set(title=regime + ': all methods and scopes on reused DEV', xlabel='Case-weighted raw teacher-score gap (lower is better)')
    axes[0, 0].legend(fontsize=8, title='Fit seed; same 36 reused DEV paths')
    dots(axes[1, 0], ('ols', 'ridge'), ('OLS', 'Ridge'), lambda a, s: fits[a, s]['standalone_adaptation_seconds'])
    axes[1, 0].set(title='All six direct-solve adaptation costs', xlabel='Seconds: extraction + design + solve + export + TRAIN validation')
    cells = sorted(records['audit']['comparisons']['cells'], key=lambda c: (c['seed'], c['regime']))
    flags = [[int(c[k][a]) for a in ARMS[:2] for k in ('later_gain_at_least_5pct', 'full_gap_nonregression')] for c in cells]
    ax = axes[1, 1]
    ax.imshow(flags, cmap=ListedColormap(('#f5d6d6', '#d6e9d8')), vmin=0, vmax=1, aspect='auto')
    for i, row in enumerate(flags):
        for j, value in enumerate(row):
            ax.text(j, i, 'PASS' if value else 'FAIL', ha='center', va='center', fontsize=9)
    ax.set(xticks=range(4), xticklabels=['Parent:\nlater >=5%', 'Parent:\nfull no worse', 'Adam74:\nlater >=5%', 'Adam74:\nfull no worse'],
           yticks=range(6), yticklabels=[f"{c['seed']} / {c['regime']}" for c in cells], title='Ridge: every paired check retained')
    decision = records['audit']['comparisons']
    fig.suptitle(f"Direct readout: {closure['status']} ({decision['passed_cells']}/6 cells)\nReused DEV diagnostic; no fresh confirmation", fontsize=17, weight='bold')
    fig.savefig(output / 'methods.png', dpi=160)
    plt.close(fig)

    def actual_loss(arm, seed):
        report = reports['train', arm, seed]
        return (report['scopes']['full']['overall']['episode_weighted_centered_mse']
                + report['prequery']['overall']['episode_weighted_centered_mse']) / 4096

    loss_rows = []
    for seed in SEEDS:
        for arm in ARMS[:4]:
            check = checks['ols' if arm not in ('ols', 'ridge') else arm, seed]
            cached = check['parent_loss'] if arm == 'pretrained' else check['adam74_loss'] if arm == 'action_residual_only' else check['data_loss']
            exported = check['exported_cached_loss'] if arm in ('ols', 'ridge') else cached
            actual = h.finite(actual_loss(arm, seed))
            loss_rows.append([seed, arm, h.fmt(cached), h.fmt(exported), h.fmt(actual), h.fmt(actual - exported)])
    def link(pin):
        return '[' + Path(pin['path']).name + '](<' + os.path.relpath(pin['path'], output) + '>)'
    def outcome(value):
        return 'PASS' if value else 'FAIL'
    lines = ['# OTTO direct-readout diagnostic', '',
        f"**{closure['status']}: ridge passed {decision['passed_cells']}/6 paired cells.**", '',
        'All five methods and three fit seeds use the same 36 previously exposed DEV paths. These paths were held out from training, but this is reused development evidence. Neither TEST nor confirmation is admitted. No autonomous performance, calibrated probabilities, statistical equivalence or architecture novelty is established.', '',
        '![Every method, fit seed, measured adaptation cost and ridge check](methods.png)', '',
        'Dots show all three fits and diamonds their equal means. Fits share evaluation paths; the dots are not independent evaluation samples or confidence intervals.', '',
        '## All fifteen DEV views', '', h.table(['Method', 'Seed', 'lambda3 full', 'lambda3 later', 'lambda4 full', 'lambda4 later'],
            [[a, s, *(h.fmt(gap(a, s, r, scope)) for r in REGIMES for scope in ('full', 'later'))] for a in ARMS for s in SEEDS]), '',
        'Raw gap is teacher score of the selected legal action minus the minimum legal teacher score. Lower is better. Full includes all nonquery actions; later includes nonquery steps >=5. Cases and collectors retain their registered weighting, including zero-support cases.', '',
        '## Every paired ridge comparison', '', h.table(['Seed', 'Setting', 'Supported later cases', 'Parent later/full', 'Adam74 later/full', 'Cell'],
            [[c['seed'], c['regime'], f"{c['supported_later_cases']}/6", *(' / '.join(outcome(c[k][a]) for k in ('later_gain_at_least_5pct', 'full_gap_nonregression')) for a in ARMS[:2]), outcome(c['passed'])] for c in cells]), '',
        'Each cell requires >=5% later-gap reduction, strict improvement, and no full-gap regression against both the parent and Adam74 residual head. OLS and full joint remain descriptive comparators; neither replaces the registered ridge candidate.', '',
        '## All six adaptation times', '', h.table(['Seed', 'Solver', 'Extract s', 'Design s', 'Solve s', 'Export s', 'Validate s', 'Standalone total s'],
            [[s, a, h.fmt(fits[a, s]['cache_metadata']['seconds']), *(h.fmt(fits[a, s][k]) for k in ('cache_design_seconds', 'solve_seconds', 'export_seconds', 'train_validation_seconds', 'standalone_adaptation_seconds'))] for s in SEEDS for a in ('ols', 'ridge')]), '',
        'Each standalone total charges the shared cache and design fully to that solver, so summing both totals double-counts shared work. Parent pretraining, historical Adam training, collection, DEV evaluation and independent audit are outside these totals. These are measured qualified CPU costs, not optimized inference-speed benchmarks.', '',
        '## Cached versus ordinary TRAIN loss', '', h.table(['Seed', 'Method', 'Float64 cached data loss', 'Exported-head cached loss', 'Ordinary float32 output loss', 'Ordinary minus exported'], loss_rows), '',
        'Loss sums the episode-weighted legal nonquery centered MSE and all-action prequery centered MSE, divided by 64². Unsupported episodes remain in the denominator 54. Ridge regularization is excluded from this data-loss comparison. Float64 cache arithmetic and ordinary float32 outputs are distinct; all twelve TRAIN parity checks passed before DEV was opened.', '',
        'OLS uses the retained numerical subspace at rcond=1e-10; ridge fixes lambda=1e-4 and penalizes all 87 contrast increment coordinates, including biases. The least-squares objective is not an action-gap optimum. No new solver or model claim is made.', '',
        '## Authenticated evidence', '', f"Closure: {link(h.descriptor(path))} (`{sha}`).", '',
        *['- ' + role.replace('_', ' ') + ': ' + link(closure[role]) + ' (`' + closure[role]['sha256'] + '`)' for role in ROLES],
        '- fit records: ' + link(fits_pin), '',
        'Both original process closures were authenticated before reporting. This renderer reads saved JSON and opaque hashes only: zero array/checkpoint decodes, model calls, solver calls, optimizer updates, teacher calls or simulator calls. Previous studies remain closed.', '']
    (output / 'README.md').write_text('\n'.join(lines))
    for pin in [*(closure[r] for r in ROLES), closure['closure_source'], fits_pin, own]:
        h.require(h.descriptor(pin['path']) == pin, 'unchanged report input or source')
    h.require(h.descriptor(path)['sha256'] == sha and h.descriptor(HELPER)['sha256'] == HELPER_PIN
              and h.descriptor(RUNNER)['sha256'] == RUNNER_PIN, 'unchanged closure and pinned helpers')
    receipt = {'version': 'otto-direct-readout-report-v1', 'status': 'completed', 'scientific_status': closure['status'],
        'closure': h.descriptor(path), 'inputs': {**{r: closure[r] for r in ROLES}, 'fits': fits_pin},
        'renderer': own, 'helper': h.descriptor(HELPER), 'runner': h.descriptor(RUNNER),
        'dev_views_displayed': 15, 'train_losses_displayed': 12, 'adaptation_times_displayed': 6, 'paired_checks_displayed': 24,
        'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0, 'solver_calls': 0,
        'optimizer_updates': 0, 'teacher_calls': 0, 'simulator_calls': 0,
        'dev_reused': True, 'fresh_dev': False, 'test_admitted': False, 'confirmation_admitted': False,
        'files': {n: h.descriptor(output / n) for n in ('README.md', 'methods.png')}}
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
