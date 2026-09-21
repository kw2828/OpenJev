"""Publish a completed, independently audited capacity screen from saved JSON only."""
import argparse
import hashlib
import importlib.util
import json
import math
import resource
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUDITOR = 'scripts/audit_otto_capacity.py'
AUDITOR_PIN = '68b6c39837c8bf58ab519bf0437a7e7ffb39c0cd66fcf2c26274786fbf3073b0'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
VERSION = 'otto-capacity-publication-v1'
KINDS, SEEDS = ('mlp8', 'mlp128', 'deep128'), (10101, 10102, 10103)
LABELS = ('MLP 8', 'MLP 128', 'MLP 128 x 3')
DESTINATIONS = {'otto-capacity-results.md': 'research/otto-capacity-results.md',
                'otto-capacity.png': 'docs/assets/otto-capacity.png',
                'otto-capacity.svg': 'docs/assets/otto-capacity.svg'}
LIMITS = {'seconds': 180, 'rss_bytes': 2*1024**3, 'output_bytes': 64*1024**2}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path, check=lambda: None):
    require(path.is_absolute() and path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'absolute regular nonsymlink input')
    result, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while block := stream.read(1024**2):
            check()
            result.update(block)
            size += len(block)
    return {'sha256': result.hexdigest(), 'bytes': size}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


class Publication:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.receipt = {'version': VERSION, 'status': 'started', 'model_calls': 0, 'simulator_calls': 0,
                        'training_calls': 0, 'published': {}, 'scope': 'Saved audited JSON; no array decoding or scientific rescoring.'}

    def check(self):
        require(self.clock.now_ns()-self.start < LIMITS['seconds']*10**9, 'publication deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
        require(rss <= LIMITS['rss_bytes'], 'publication RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'publication output cap')

    def authenticate(self):
        args = self.args
        require(digest(args.plan, self.check)['sha256'] == args.plan_sha256, 'external plan pin')
        plan = json.loads(args.plan.read_text())
        require(plan['version'] == 'otto-capacity-v1' and plan['mode'] == 'study'
                and plan['status'] == 'frozen_before_execution', 'full frozen study')
        require(plan['sources'][AUDITOR] == AUDITOR_PIN == digest(ROOT/AUDITOR, self.check)['sha256'], 'held independent auditor')
        audit_module = load(ROOT/AUDITOR, '_capacity_publisher_auth')
        for name, pin in plan['sources'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'contained source')
            require(digest(ROOT/name, self.check)['sha256'] == pin, 'unchanged prospective source')
        worker = audit_module.manifest(args.run, args.receipt_sha256, audit_module.payload_names(), self.check)
        audit = audit_module.manifest(args.audit, args.audit_receipt_sha256, {'started.json', 'summary.json'}, self.check)
        require(digest(args.terminal, self.check)['sha256'] == args.terminal_sha256, 'external parent terminal pin')
        require(worker['version'] == plan['version'] and worker['mode'] == 'study'
                and worker['plan_sha256'] == args.plan_sha256 and worker['sources'] == plan['sources']
                and worker['inputs'] == plan['inputs'] and worker['limits'] == plan['limits']
                and worker['completed_fits'] == 9 and worker['parity_passed'] is True and worker['pending'] == []
                and all(worker[k] == 0 for k in ('external_model_calls', 'native_steps', 'native_resets'))
                and all(v['attempted'] == v['returned'] for v in worker['calls'].values())
                and worker['calls']['optimizer_update']['returned'] == 31680, 'complete bound worker')
        require(audit['version'] == 'otto-capacity-saved-audit-v1' and audit['agreement'] is True
                and audit['source']['sha256'] == AUDITOR_PIN and audit['plan_sha256'] == args.plan_sha256
                and audit['worker_sha256'] == args.receipt_sha256 and audit['terminal_sha256'] == args.terminal_sha256
                and audit['producer_source_sha256'] == plan['sources']['scripts/study_otto_capacity.py'], 'matching successful independent audit')
        request = audit_module.read(args.audit/'started.json')['request']
        for key in ('plan', 'plan_sha256', 'run', 'receipt_sha256', 'terminal', 'terminal_sha256'):
            require(request[key] == str(getattr(args, key)), 'audit request identity')
        require(request['output'] == str(args.audit), 'actual audit location')
        audit_module.check_terminal(plan, args.plan, worker, audit_module.read(args.run/'started.json'),
                                    audit_module.read(args.terminal), args.run, self.check)
        result, producer = audit_module.read(args.audit/'summary.json'), audit_module.read(args.run/'summary.json')
        require(result['agreement'] is True and result['audit_version'] == audit['version']
                and result['independent_readout_calls'] == 252 and result['independent_readout_rows'] == 60444
                and result['learned_architecture_advantage_established'] is False, 'complete audit coverage')
        for key, value in producer.items():
            if key != 'scope':
                audit_module.close(value, result[key], 'producer/audit '+key)
        fits = [json.loads(line) for line in (args.run/'fits.jsonl').read_text().splitlines()]
        require([r['fit_id'] for r in fits] == [f'{kind}@{seed}' for seed in SEEDS for kind in KINDS], 'all nine ordered fits')
        for fit in fits:
            audit_module.close(fit['metrics'], result['metrics'][fit['fit_id']], 'fit metrics')
            require(fit['fit_seconds'] == result['training_costs'][fit['fit_id']], 'fit cost join')
        self.receipt['inputs'] = {key: {'path': str(path), **digest(path, self.check)} for key, path in {
            'plan': args.plan, 'worker': args.run/'receipt.json', 'terminal': args.terminal,
            'audit': args.audit/'receipt.json', 'summary': args.audit/'summary.json', 'fits': args.run/'fits.jsonl'}.items()}
        return result, fits, worker


def display_rows(summary, fits):
    expected = [f'{kind}.{seed}.{metric}' for kind in KINDS[1:] for seed in SEEDS for metric in ('train_excess', 'valid_mse')]
    require([row['name'] for row in summary['checks']] == expected, 'all twelve fixed conditions')
    for row in summary['checks']:
        require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in ('value', 'threshold'))
                and type(row['passes']) is bool and row['passes'] == (row['value'] <= row['threshold']), 'unrounded inequality identity')
    require(summary['capacity_screen_passed'] is all(r['passes'] for r in summary['checks']), 'overall decision')
    floor = summary['alias_floor']['irreducible_mse_normalized']
    require(type(floor) in (float, int) and math.isfinite(floor) and floor >= 0, 'finite empirical floor')
    rows = []
    for fit in fits:
        train, valid = summary['metrics'][fit['fit_id']]['train'], summary['metrics'][fit['fit_id']]['valid']
        require(train['rows'] == 5589 and valid['rows'] == 1109, 'complete scoring populations')
        row = {'fit_id': fit['fit_id'], 'kind': fit['kind'], 'seed': fit['seed'],
               'parameter_count': fit['parameter_count'], 'fit_seconds': fit['fit_seconds'],
               'train_mse': train['mse_normalized'], 'train_excess': train['mse_normalized']-floor,
               'valid_mse': valid['mse_normalized'], 'train_mae_moves': train['mae_physical'],
               'valid_mae_moves': valid['mae_physical'], 'train_negative': train['negative_predictions'],
               'valid_negative': valid['negative_predictions']}
        require(all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float))), 'finite display fields')
        rows.append(row)
    return rows


def plot(rows, summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    figure, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = [('train_excess', 'TRAIN MSE minus empirical alias floor'),
               ('valid_mse', 'VALID MSE'), ('fit_seconds', 'Paid fit time (seconds)')]
    colors, markers = ('#2166ac', '#bd4e20', '#237d64'), ('o', 's', '^')
    for ax, (key, title) in zip(axes, metrics, strict=True):
        for seed, color, marker in zip(SEEDS, colors, markers, strict=True):
            values = [next(r[key] for r in rows if r['kind'] == kind and r['seed'] == seed) for kind in KINDS]
            ax.plot(range(3), values, color=color, marker=marker, linewidth=1.2, markersize=6, alpha=.85)
        means = [math.fsum(r[key] for r in rows if r['kind'] == kind)/3 for kind in KINDS]
        ax.plot(range(3), means, color='#16202c', marker='_', markersize=20, linestyle='none', markeredgewidth=3)
        ax.set_xticks(range(3), LABELS)
        ax.set_title(title, fontsize=12, pad=12)
        ax.axhline(0, color='#89949c', linewidth=.7)
        ax.grid(axis='y', alpha=.18)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))
        lower, upper = ax.get_ylim()
        ax.set_ylim(min(lower, 0), max(upper, 0))
    passed = sum(c['passes'] for c in summary['checks'])
    status = 'PASS' if summary['capacity_screen_passed'] else 'FAIL'
    figure.suptitle('Fixed-data value regression capacity', fontsize=20, fontweight='bold', y=.98)
    figure.text(.5, .885, f'{status}: {passed}/12 conditions passed | 5,589 TRAIN + 1,109 exposed VALID rows | no action evaluation', ha='center', fontsize=11)
    handles = [Line2D([], [], color=c, marker=m, label=f'Seed {s}') for s, c, m in zip(SEEDS, colors, markers, strict=True)]
    handles.append(Line2D([], [], color='#16202c', marker='_', markersize=16, linestyle='none', label='Equal-seed mean'))
    figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .065), ncol=4, frameon=False)
    figure.text(.5, .025, 'Lower is better. Signed TRAIN excess is preserved. Paired seed lines and means are descriptive, not confidence intervals.', ha='center', fontsize=10)
    figure.subplots_adjust(top=.77, bottom=.25, left=.07, right=.985, wspace=.33)
    figure.savefig(out/'otto-capacity.png', dpi=220)
    figure.savefig(out/'otto-capacity.svg')
    plt.close(figure)


def report(rows, summary, worker, args):
    passed = sum(c['passes'] for c in summary['checks'])
    status = 'PASS' if summary['capacity_screen_passed'] else 'FAIL'
    floor = summary['alias_floor']
    train_passes = sum(c['passes'] for c in summary['checks'] if c['name'].endswith('.train_excess'))
    valid_passes = sum(c['passes'] for c in summary['checks'] if c['name'].endswith('.valid_mse'))
    means = {kind: {key: math.fsum(r[key] for r in rows if r['kind'] == kind)/3
                    for key in ('valid_mse', 'train_excess')} for kind in KINDS}
    def change(kind, key):
        base, value = means['mlp8'][key], means[kind][key]
        if base <= 0:
            return 'not expressed as a percentage because the narrow mean is nonpositive'
        return f'{100*abs(value-base)/base:.2f}% {"lower" if value < base else "higher" if value > base else "unchanged"}'
    deep_improved = sum(next(r['valid_mse'] for r in rows if r['kind'] == 'deep128' and r['seed'] == seed)
                        < next(r['valid_mse'] for r in rows if r['kind'] == 'mlp8' and r['seed'] == seed) for seed in SEEDS)
    text = ['# Fixed-data scalar capacity screen', '',
        f'**{status}: {passed}/12 frozen conditions passed.** TRAIN excess: {train_passes}/6; VALID improvement: {valid_passes}/6.', '',
        (f'Against the fresh narrow baseline, mean VALID MSE is {change("mlp128", "valid_mse")} for MLP 128 '
         f'and {change("deep128", "valid_mse")} for the three-layer network. Its mean TRAIN excess is '
         f'{change("deep128", "train_excess")}. Its VALID MSE improves in {deep_improved}/3 paired seeds. '
         'These equal-seed averages do not replace the fixed per-seed criteria.'), '',
        ('This study compares fresh ordinary-network capacity on identical Monte Carlo teacher-return labels. '
        'It includes no autonomous searches, observation-branch scoring or new collection. It establishes no recurrent, connectome or novel-architecture advantage.'), '',
        '![All nine fits, scalar errors and paid fitting costs](../docs/assets/otto-capacity.png)', '',
        ('All models use the same 5,589 TRAIN states from 192 completed teacher episodes and 1,109 VALID states from 48 episodes, '
        'with uniform row MSE, three paired seeds, 80 epochs and fixed final checkpoints. VALID is exposed research validation. '
        'The initialization recipe is new; the freshly trained width-eight control is the relevant capacity baseline.'), '',
        (f'The empirical TRAIN feature-alias floor is **{floor["irreducible_mse_normalized"]:.12g}** normalized MSE. '
        f'There are {floor["unique_groups"]:,} byte-identical-input groups, {floor["duplicate_groups"]:,} duplicate groups '
        f'containing {floor["duplicate_rows"]:,} rows, and {floor["conflicting_groups"]:,} groups with conflicting returns. '
        'This is a finite-cohort bound for these cached inputs, not a population uncertainty estimate.'), '',
        '| Fit | TRAIN MSE | TRAIN excess | VALID MSE | TRAIN MAE, moves | VALID MAE, moves |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f'| {r["fit_id"]} | '+ ' | '.join(f'{r[k]:.9g}' for k in ('train_mse', 'train_excess', 'valid_mse', 'train_mae_moves', 'valid_mae_moves'))+' |')
    text += ['', 'MSE is in normalized return units `(T-t)/64`; MAE is in movement units. Signed excess is neither clipped nor given an epsilon allowance.', '',
             '| Fit | Parameters | TRAIN negative predictions | VALID negative predictions | Paid fit seconds |', '|---|---:|---:|---:|---:|']
    text += [f'| {r["fit_id"]} | {r["parameter_count"]:,} | {r["train_negative"]} | {r["valid_negative"]} | {r["fit_seconds"]:.6f} |' for r in rows]
    text += ['', (f'Total paid fit time: {math.fsum(r["fit_seconds"] for r in rows):.6f} s. Complete worker duration: {worker["wall_seconds"]:.6f} s. '
             'Fit time includes initialization, optimization and checkpoint export/publication. It excludes final scalar parity and saved-prediction diagnostics. '
             'The worker also pays authentication, preparation, those diagnostics and output work. Equal optimizer updates do not imply equal computation; these are single-run hardware-specific measurements.'), '',
             '| Condition | Actual | Required maximum | Result |', '|---|---:|---:|---|']
    text += [f'| {c["name"]} | {c["value"]:.17g} | {c["threshold"]:.17g} | {"Pass" if c["passes"] else "Fail"} |' for c in summary['checks']]
    text += ['', ('All twelve direct float64 conditions are required. Both wider families and all seeds are retained. '
             'A pass motivates a separately frozen autonomous comparison, not a competence claim. A failure rejects this fixed recipe; it does not rule out all higher-capacity models.'), '',
             ('This is distinct from the earlier [Bellman-versus-Monte-Carlo continuation](otto-bellman-control-results.md), '
             'which used the same small architecture, different targets and autonomous evaluation, passed 19/42 conditions and failed its overall rule. '
             'That decision and the original scalar trial\'s 0/54 failure remain unchanged. Smaller scalar error need not improve action ranking.'), '',
             ('The independent audit reconstructed every cached feature/target and replayed all final scalar predictions: '
             f'{summary["independent_readout_calls"]:,} independent readouts covering {summary["independent_readout_rows"]:,} rows including parity witnesses. '
             'Optimizer execution, original Torch parity and timings remain authenticated execution evidence.'), '',
             '[Protocol](otto-capacity-protocol.md) | [Conditional next-step review](../output/otto-capacity-v1/next-experiment-review.md)', '',
             '| Provenance | Bound artifact | SHA-256 prefix |', '|---|---|---|']
    for label, path, pin in (('Plan', args.plan, args.plan_sha256), ('Worker', args.run/'receipt.json', args.receipt_sha256),
                             ('Independent audit', args.audit/'receipt.json', args.audit_receipt_sha256), ('Parent terminal', args.terminal, args.terminal_sha256)):
        text.append(f'| {label} | [Record](../{path.relative_to(ROOT)}) | `{pin[:16]}` |')
    text += ['', (f'[Full audit summary](../{(args.audit/"summary.json").relative_to(ROOT)}) | '
             f'[Plot values and exact criteria](../{(args.output/"plotted-values.json").relative_to(ROOT)}) | '
             f'[Publication receipt](../{(args.output/"receipt.json").relative_to(ROOT)}) | '
             '[Publisher source](../output/otto-capacity-v1/report-support/publish.py)'), '',
             ('[Complete evidence archive](https://github.com/kw2828/OpenJev/releases/tag/otto-capacity-v1). '
              'The archive preserves this study and its qualification/provenance records. Historical input caches remain '
              'separate dependencies from the [original scalar release](https://github.com/kw2828/OpenJev/releases/tag/otto-return-value-v1).'), '']
    return '\n'.join(text)


def execute(args):
    require(args.output.is_absolute() and not args.output.exists()
            and not any(p.is_symlink() for p in args.output.parents), 'exclusive publication output')
    args.output.mkdir(parents=True, exist_ok=False)
    publication = Publication(args)
    try:
        require(digest(ROOT/CLOCK)['sha256'] == CLOCK_PIN, 'qualified clock')
        publication.clock = load(ROOT/CLOCK, '_capacity_publication_clock').SuspendClock()
        publication.start = publication.clock.now_ns()
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('publication wall alarm')))
        signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
        publication.receipt.update(source=digest(Path(__file__)), helper=digest(ROOT/AUDITOR), limits=LIMITS,
                                   request={k: str(v) for k, v in vars(args).items()})
        summary, fits, worker = publication.authenticate()
        rows = display_rows(summary, fits)
        write(args.output/'plotted-values.json', {'rows': rows, 'alias_floor': summary['alias_floor'],
              'checks': summary['checks'], 'capacity_screen_passed': summary['capacity_screen_passed'],
              'worker_seconds': worker['wall_seconds'], 'means': {kind: {metric: math.fsum(r[metric] for r in rows if r['kind'] == kind)/3
                for metric in ('train_excess', 'valid_mse', 'fit_seconds')} for kind in KINDS}})
        (args.output/'otto-capacity-results.md').write_text(report(rows, summary, worker, args))
        publication.check()
        plot(rows, summary, args.output)
        publication.check()
        publication.authenticate()
        require(digest(Path(__file__), publication.check) == publication.receipt['source'], 'publisher source unchanged')
        require(all(not (ROOT/target).exists() and not any(p.is_symlink() for p in (ROOT/target).parents)
                    for target in DESTINATIONS.values()), 'new report/assets only; no overwrite')
        for name, destination in DESTINATIONS.items():
            target = ROOT/destination
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write((args.output/name).read_bytes())
            publication.receipt['published'][destination] = digest(target, publication.check)
            require(publication.receipt['published'][destination] == digest(args.output/name, publication.check), 'published bytes match staged artifact')
        publication.receipt.update(status='completed', elapsed_seconds=(publication.clock.now_ns()-publication.start)/1e9,
                                   files={p.name: digest(p, publication.check) for p in args.output.iterdir()})
        write(args.output/'receipt.json', publication.receipt)
        publication.check()
        signal.setitimer(signal.ITIMER_REAL, 0)
        return publication.receipt
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        publication.receipt.update(status='failed', error=repr(error))
        try:
            if (args.output/'receipt.json').exists():
                (args.output/'receipt.json').rename(args.output/'invalid-completed-receipt.json')
            write(args.output/'failed.json', publication.receipt)
        except BaseException as secondary:  # noqa: BLE001 - Preserve primary publication failure.
            error.add_note(f'Failure publication: {secondary!r}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'run', 'terminal', 'audit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('plan-sha256', 'receipt-sha256', 'terminal-sha256', 'audit-receipt-sha256'):
        parser.add_argument('--'+name, required=True)
    execute(parser.parse_args())
