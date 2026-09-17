"""Revalidate and publish compute-control or mate-curriculum chess evidence.

Publication reruns the matching frozen report, never model scoring. Original
evidence bytes are preserved in a deterministic archive. Rendering reads only
published summaries after their archive, receipt and checkpoint hashes pass audit.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = 'execution-and-report.tar.gz'
MATE_NAMES = {f'{arm}-{seed}' for arm in ('single', 'set') for seed in (17, 29, 43)}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    def invalid(_):
        raise ValueError('Nonfinite JSON number')

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return decode(Path(path).read_bytes())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def copy_new(source, destination):
    with Path(source).open('rb') as incoming, Path(destination).open('xb') as outgoing:
        while block := incoming.read(1024*1024):
            outgoing.write(block)


def close(actual, expected):
    if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(
            actual, expected, abs_tol=1e-10, rel_tol=1e-10):
        raise ValueError('Summary arithmetic is inconsistent')


def load_study(kind):
    if kind not in ('compute', 'mate'):
        raise ValueError('Unknown refinement study kind')
    spec = importlib.util.spec_from_file_location('_refinement_'+kind, ROOT/f'scripts/chess_{kind}_study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_summary(kind, plan, summary):
    if kind == 'compute':
        if (summary['status'] != 'completed' or summary['novelty_established'] is not False or
                summary['elo_estimate'] is not None):
            raise ValueError('Expected a scoped completed compute summary')
        expected = {(config['id'], split) for config in plan['configurations'] for split in ('dev', 'shift')}
        records = summary['configurations']
        if len(records) != len(expected) or {(r['configuration'], r['split']) for r in records} != expected:
            raise ValueError('Compute summary does not contain the full configuration panel')
        configs = {config['id']: config for config in plan['configurations']}
        for row in records:
            config = configs[row['configuration']]
            if (any(row[key] != config[key] for key in ('mode', 'seed', 'variant')) or
                    row['examples'] != len(plan['panels'][row['split']])):
                raise ValueError('Compute summary configuration or panel count mismatch')
            for field in ('agreement', 'latency_mean_ms', 'latency_p50_ms', 'latency_p95_ms'):
                value = row[field]
                if not math.isfinite(value) or value < 0 or (field == 'agreement' and value > 1):
                    raise ValueError('Invalid compute summary metric')
        return summary
    arms, seeds = ('frozen', 'single', 'set'), (17, 29, 43)
    expected = {f'{arm}-{seed}' for arm in arms for seed in seeds}
    splits = ('mate_dev', 'mate_confirm', 'dev', 'shift')
    if set(summary['results']) != expected or set(summary['mean_accuracy']) != set(arms):
        raise ValueError('Mate summary lacks the full nine-model panel')
    for result in summary['results'].values():
        if set(result['metrics']) != set(splits):
            raise ValueError('Mate summary lacks a required evaluation panel')
        for row in result['metrics'].values():
            if (type(row['examples']) is not int or row['examples'] <= 0 or
                    type(row['correct']) is not int or not 0 <= row['correct'] <= row['examples']):
                raise ValueError('Invalid mate example or success count')
            close(row['accuracy'], row['correct']/row['examples'])
    for arm in arms:
        if set(summary['mean_accuracy'][arm]) != set(splits):
            raise ValueError('Mate mean split membership mismatch')
        for split in splits:
            values = [summary['results'][f'{arm}-{seed}']['metrics'][split]['accuracy'] for seed in seeds]
            close(summary['mean_accuracy'][arm][split], math.fsum(values)/3)
    for split in splits:
        close(summary['set_minus_frozen'][split],
              summary['mean_accuracy']['set'][split]-summary['mean_accuracy']['frozen'][split])
        close(summary['set_minus_single'][split],
              summary['mean_accuracy']['set'][split]-summary['mean_accuracy']['single'][split])
    gate = plan['protocol']['gate']
    checks = {'targeted_skill': all(summary['set_minus_frozen'][split] >= gate['mate_gain_over_frozen']
                                    for split in ('mate_dev', 'mate_confirm')),
              'ordinary_retention': all(summary['set_minus_frozen'][split] >= -gate['max_ordinary_mean_drop']
                                       for split in ('dev', 'shift')),
              'set_objective': all(summary['set_minus_single'][split] >= gate['set_gain_over_single']
                                   for split in ('mate_dev', 'mate_confirm'))}
    if summary['gates'] != checks or summary['new_training_engine_calls'] != 0:
        raise ValueError('Mate frozen gates or training engine-call count mismatch')
    return summary


def inventory(execution, report, kind):
    execution, report = Path(execution), Path(report)
    if execution.is_symlink() or not execution.is_dir():
        raise ValueError('Execution must be a real directory')
    paths = {}
    roots = [('execution', execution)] + ([('report', report)] if kind == 'compute' else [])
    for prefix, directory in roots:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError('Report must be a real directory')
        for path in sorted(directory.rglob('*')):
            if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                raise ValueError('Evidence contains a link or special file')
            if path.is_file():
                if path.name == 'failed.json':
                    raise ValueError('Failed evidence cannot be published')
                paths[f'{prefix}/{path.relative_to(directory).as_posix()}'] = path
    if kind == 'mate':
        if report.is_symlink() or not report.is_file():
            raise ValueError('Mate report must be a real summary file')
        paths['report/summary.json'] = report
    return paths, {name: {'sha256': sha(path), 'size': path.stat().st_size} for name, path in paths.items()}


def make_archive(path, paths, members):
    with (Path(path).open('xb') as raw,
          gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed,
          tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive):
        for name, metadata in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = metadata['size'], 0o644, 0
            with paths[name].open('rb') as stream:
                archive.addfile(info, stream)


def inspect_archive(path, expected):
    actual, documents = {}, {}
    with tarfile.open(path, mode='r|gz') as archive:
        for member in archive:
            name = member.name
            parts = PurePosixPath(name).parts
            if (not member.isfile() or name in actual or name not in expected or not parts or
                    parts[0] not in ('execution', 'report') or '..' in parts or str(PurePosixPath(name)) != name):
                raise ValueError('Unexpected, duplicated or unsafe archive member')
            if parts[-1] == 'failed.json':
                raise ValueError('An archived failure receipt prevents publication')
            stream = archive.extractfile(member)
            digest, chunks = hashlib.sha256(), []
            keep = name.endswith('/completed.json') or name in ('execution/started.json', 'report/summary.json')
            while chunk := stream.read(1024*1024):
                digest.update(chunk)
                if keep:
                    chunks.append(chunk)
            actual[name] = {'sha256': digest.hexdigest(), 'size': member.size}
            if actual[name] != expected[name]:
                raise ValueError('Archive payload differs from manifest')
            if keep:
                documents[name] = decode(b''.join(chunks))
    if actual != expected:
        raise ValueError('Archive member set differs from manifest')
    return documents


def validate_chain(kind, plan_hash, summary, members, documents):
    if not {'execution/started.json', 'execution/completed.json', 'report/summary.json'} <= set(members):
        raise ValueError('Archive lacks the execution completion chain')
    complete = documents['execution/completed.json']
    if (complete['status'] != 'completed' or complete['plan_sha256'] != plan_hash or
            documents['execution/started.json']['plan_sha256'] != plan_hash or
            summary['plan_sha256'] != plan_hash or
            summary['execution_receipt_sha256'] != members['execution/completed.json']['sha256'] or
            documents['report/summary.json'] != summary):
        raise ValueError('Published summary and execution completion chain differ')
    for name, receipt in documents.items():
        if not name.endswith('/completed.json'):
            continue
        if receipt['status'] != 'completed':
            raise ValueError('Archived receipt is incomplete')
        prefix = str(PurePosixPath(name).parent)
        for relative, digest in receipt.get('files', {}).items():
            target = prefix+'/'+relative
            if target not in members or members[target]['sha256'] != digest:
                raise ValueError('Archived receipt references missing or changed evidence')
    if kind == 'compute':
        report = documents['report/completed.json']
        if (report['plan_sha256'] != plan_hash or report['summary_sha256'] != members['report/summary.json']['sha256'] or
                report['execution_receipt_sha256'] != members['execution/completed.json']['sha256']):
            raise ValueError('Compute report receipt is unbound')
        if complete['decisions'] != sum(row['examples'] for row in summary['configurations']):
            raise ValueError('Compute decision count differs from the complete summary')
    else:
        expected = MATE_NAMES | {f'frozen-{seed}' for seed in (17, 29, 43)}
        if set(complete['fits']) != expected:
            raise ValueError('Mate execution requires all six trained and three frozen fits')
        if (members.get('execution/schedule.json', {}).get('sha256') != complete['schedule_sha256']):
            raise ValueError('Mate training schedule binding differs from the original')
        for name, digest in complete['fits'].items():
            path = f'execution/{name}/completed.json'
            if path not in members or members[path]['sha256'] != digest:
                raise ValueError('Mate fit receipt is missing or changed')
            arm, seed = name.split('-')
            fit = documents[path]
            if fit['arm'] != arm or fit['seed'] != int(seed) or fit['plan_sha256'] != plan_hash:
                raise ValueError('Mate fit identity differs from its receipt')


def publish(kind, plan, execution, report, out, models=None):
    plan, execution, report, out = map(Path, (plan, execution, report, out))
    models = ROOT/'models/chess-mate-v1' if models is None else Path(models)
    if kind not in ('compute', 'mate'):
        raise ValueError('Unknown refinement study kind')
    if out.exists() or (kind == 'mate' and models.exists()):
        raise FileExistsError('Publication or mate weights directory already exists')
    frozen = read(plan)
    prepared = read(plan.parent/'prepared.json')
    if prepared['status'] != 'prepared' or prepared['plan_sha256'] != sha(plan):
        raise ValueError('Prepared receipt is not bound to this plan')
    if kind == 'mate' and sha(plan.parent/'selection.json') != frozen['selection_sha256']:
        raise ValueError('Mate selection differs from its frozen plan')
    summary_path = report/'summary.json' if kind == 'compute' else report
    summary = validate_summary(kind, frozen, read(summary_path))
    paths, members = inventory(execution, report, kind)
    study = load_study(kind)
    with tempfile.TemporaryDirectory(prefix=f'openjev-{kind}-publication-') as temporary:
        target = Path(temporary)/('report' if kind == 'compute' else 'summary.json')
        recomputed = study.report(plan, execution, target)
    if summary != recomputed:
        raise ValueError('Supplied summary does not reproduce from the frozen report validator')
    if members != inventory(execution, report, kind)[1]:
        raise ValueError('Source evidence changed during revalidation')
    out.mkdir(parents=True, exist_ok=False)
    try:
        source_files = {'plan.json': plan, 'prepared.json': plan.parent/'prepared.json', 'summary.json': summary_path}
        if kind == 'mate':
            source_files['selection.json'] = plan.parent/'selection.json'
        for name, path in source_files.items():
            copy_new(path, out/name)
        make_archive(out/ARCHIVE, paths, members)
        if members != inventory(execution, report, kind)[1]:
            raise ValueError('Source evidence changed while archiving')
        documents = inspect_archive(out/ARCHIVE, members)
        validate_chain(kind, sha(plan), summary, members, documents)
        weights, model_files = {}, {}
        if kind == 'mate':
            models.mkdir(parents=True, exist_ok=False)
            copy_new(ROOT/'LICENSE', models/'LICENSE')
            card = ('# OpenJev mate curriculum weights\n\nSix original fine-tuned OpenJev predict checkpoints. '
                    'All final single-target and mating-set fits, seeds 17, 29 and 43, are included.\n\n'
                    'Experimental: mate-in-one accuracy increased, but ordinary move agreement declined. '
                    'These checkpoints are not promoted as stronger general chess players.\n\n'
                    'License: [MIT](LICENSE). Lichess puzzle source data are CC0. '
                    'These weights do not establish Elo or architectural novelty.\n\n'
                    f'Training plan SHA-256: `{sha(plan)}`. Exact original weights are also retained in '
                    'the published execution archive.\n')
            with (models/'README.md').open('x') as stream:
                stream.write(card)
            for name in sorted(MATE_NAMES):
                destination = models/name/'weights.pt'
                destination.parent.mkdir()
                copy_new(execution/name/'weights.pt', destination)
                original = members[f'execution/{name}/weights.pt']
                if sha(destination) != original['sha256'] or destination.stat().st_size != original['size']:
                    raise ValueError('Copied mate checkpoint differs from the original')
                weights[name] = {'path': f'{name}/weights.pt', **original, 'license': 'MIT', 'license_file': 'LICENSE'}
            model_files = {name: {'sha256': sha(models/name), 'size': (models/name).stat().st_size}
                           for name in ('LICENSE', 'README.md')}
        manifest = {'format': 'openjev-chess-refinement-publication-v1', 'kind': kind,
                    'plan_sha256': sha(out/'plan.json'),
                    'files': {name: {'sha256': sha(out/name), 'size': (out/name).stat().st_size}
                              for name in source_files},
                    'archive': {'path': ARCHIVE, 'sha256': sha(out/ARCHIVE), 'size': (out/ARCHIVE).stat().st_size},
                    'members': members, 'member_count': len(members),
                    'raw_bytes': sum(row['size'] for row in members.values()),
                    'weights': weights, 'model_files': model_files}
        write_new(out/'manifest.json', manifest)
        write_new(out/'completed.json', {'status': 'completed', 'kind': kind,
                                        'plan_sha256': sha(out/'plan.json'), 'summary_sha256': sha(out/'summary.json'),
                                        'manifest_sha256': sha(out/'manifest.json'), 'publisher_sha256': sha(__file__),
                                        'source_report_reproduced': True, 'archive_roundtrip_verified': True,
                                        'new_model_scoring': False, 'published_weights': len(weights)})
        return audit(out, models=models)
    except Exception as exc:
        write_new(out/'failed.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        raise


def audit(out, models=None):
    out = Path(out)
    models = ROOT/'models/chess-mate-v1' if models is None else Path(models)
    receipt, manifest = read(out/'completed.json'), read(out/'manifest.json')
    kind = manifest['kind']
    if (kind not in ('compute', 'mate') or receipt['status'] != 'completed' or receipt['kind'] != kind or
            (out/'failed.json').exists() or receipt['manifest_sha256'] != sha(out/'manifest.json') or
            receipt['plan_sha256'] != sha(out/'plan.json') or manifest['plan_sha256'] != sha(out/'plan.json') or
            receipt['summary_sha256'] != sha(out/'summary.json') or receipt['source_report_reproduced'] is not True or
            receipt['archive_roundtrip_verified'] is not True or receipt['new_model_scoring'] is not False):
        raise ValueError('Publication completion chain is invalid')
    required = {'plan.json', 'prepared.json', 'summary.json'} | ({'selection.json'} if kind == 'mate' else set())
    if set(manifest['files']) != required:
        raise ValueError('Published source document membership is incomplete')
    for name, metadata in manifest['files'].items():
        path = out/name
        if path.is_symlink() or metadata != {'sha256': sha(path), 'size': path.stat().st_size}:
            raise ValueError('Published document differs from its original')
    plan, summary = read(out/'plan.json'), read(out/'summary.json')
    prepared = read(out/'prepared.json')
    if prepared['status'] != 'prepared' or prepared['plan_sha256'] != sha(out/'plan.json'):
        raise ValueError('Published prepared receipt is unbound')
    if kind == 'mate' and sha(out/'selection.json') != plan['selection_sha256']:
        raise ValueError('Published mate selection is unbound')
    validate_summary(kind, plan, summary)
    archive = manifest['archive']
    if (archive['path'] != ARCHIVE or (out/ARCHIVE).is_symlink() or
            archive['sha256'] != sha(out/ARCHIVE) or archive['size'] != (out/ARCHIVE).stat().st_size):
        raise ValueError('Compressed evidence archive changed')
    documents = inspect_archive(out/ARCHIVE, manifest['members'])
    if (manifest['member_count'] != len(manifest['members']) or
            manifest['raw_bytes'] != sum(row['size'] for row in manifest['members'].values()) or
            manifest['members']['report/summary.json']['sha256'] != sha(out/'summary.json')):
        raise ValueError('Archive totals or original summary binding differ')
    validate_chain(kind, sha(out/'plan.json'), summary, manifest['members'], documents)
    if set(manifest['weights']) != (MATE_NAMES if kind == 'mate' else set()):
        raise ValueError('Published checkpoint membership is incomplete')
    if receipt['published_weights'] != len(manifest['weights']):
        raise ValueError('Published checkpoint count is inconsistent')
    for name, metadata in manifest['weights'].items():
        relative = f'{name}/weights.pt'
        path = models/relative
        original = manifest['members'][f'execution/{name}/weights.pt']
        if (metadata['path'] != relative or metadata['sha256'] != original['sha256'] or
                metadata['size'] != original['size'] or metadata['license'] != 'MIT' or
                metadata['license_file'] != 'LICENSE' or path.is_symlink() or
                sha(path) != original['sha256'] or path.stat().st_size != original['size']):
            raise ValueError('Published checkpoint does not match archived original')
    if set(manifest['model_files']) != ({'LICENSE', 'README.md'} if kind == 'mate' else set()):
        raise ValueError('Published checkpoint license or model card is missing')
    for name, metadata in manifest['model_files'].items():
        path = models/name
        if path.is_symlink() or metadata != {'sha256': sha(path), 'size': path.stat().st_size}:
            raise ValueError('Checkpoint license or model card changed')
    return {'status': 'verified', 'kind': kind, 'members': manifest['member_count'],
            'plan_sha256': sha(out/'plan.json'), 'summary_sha256': sha(out/'summary.json'),
            'archive_sha256': sha(out/ARCHIVE), 'published_weights': len(manifest['weights'])}


def figure(compute, mate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.hashsalt': 'chess-refinement-v1'})
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.8), gridspec_kw={'width_ratios': [1.25, 1.25, 1]})
    fig.patch.set_facecolor('#fbfcfe')
    fig.subplots_adjust(left=.055, right=.98, top=.73, bottom=.28, wspace=.25)
    fig.text(.055, .93, 'OpenJev: extra computation and targeted chess training', fontsize=22, weight='bold')
    fig.text(.055, .86, 'Every seed is shown; bars show three-seed means. Separate experiments and evaluation panels.',
             fontsize=11, color='#506272')
    variants = ('policy_d4', 'policy_d8', 'policy_d16', 'successor_2_d4', 'successor_4_d4', 'successor_all_d4')
    labels = ('4 passes\nbaseline', '8 passes', '16 passes', 'Top 2\nsuccessors', 'Top 4\nsuccessors', 'All legal\nsuccessors')
    lookup = {(r['variant'], r['seed'], r['split']): r for r in compute['configurations'] if r['mode'] == 'predict'}
    colors = {'dev': '#238783', 'shift': '#a56b38'}
    jitter = (-.05, 0, .05)
    for x, variant in enumerate(variants):
        for offset, split, marker in ((-.18, 'dev', 'o'), (.18, 'shift', 's')):
            for ax, field, scale in ((axes[0], 'agreement', 100), (axes[1], 'latency_p50_ms', 1)):
                values = [lookup[variant, seed, split][field]*scale for seed in (17, 29, 43)]
                ax.bar(x+offset, math.fsum(values)/3, width=.31, color=colors[split], alpha=.8, zorder=2)
                ax.scatter([x+offset+delta for delta in jitter], values, marker=marker,
                           s=22, facecolor='white', edgecolor='#223849', zorder=3)
    for ax in axes[:2]:
        ax.set_xticks(range(len(variants)), labels, fontsize=8)
    axes[0].set_title('Predict model: teacher-move agreement', loc='left', fontsize=11)
    axes[0].set_ylabel('Agreement (%)')
    axes[1].set_title('Corresponding CPU decision cost', loc='left', fontsize=11)
    axes[1].set_ylabel('Per-seed median latency (ms, log scale)')
    axes[1].set_yscale('log')
    for x, arm in enumerate(('frozen', 'single', 'set')):
        for offset, split, marker, color in ((-.18, 'mate_dev', 'o', '#238783'),
                                            (.18, 'mate_confirm', 's', '#a56b38')):
            values = [mate['results'][f'{arm}-{seed}']['metrics'][split]['accuracy']*100 for seed in (17, 29, 43)]
            axes[2].bar(x+offset, math.fsum(values)/3, width=.31, color=color, alpha=.8, zorder=2)
            axes[2].scatter([x+offset+delta for delta in jitter], values, marker=marker,
                            s=22, facecolor='white', edgecolor='#223849', zorder=3)
    axes[2].set_xticks(range(3), ('Frozen', 'Single target', 'Mating set'), fontsize=9)
    axes[2].set_title('Mate-in-one: any legal mating move', loc='left', fontsize=11)
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_ylim(0, 100)
    for ax in axes:
        ax.set_facecolor('#fbfcfe')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    legends = [Line2D([], [], color=colors['dev'], marker='o', label='Compute: development / Mate: development'),
               Line2D([], [], color=colors['shift'], marker='s', label='Compute: shifted / Mate: confirmation')]
    fig.legend(handles=legends, loc='lower left', bbox_to_anchor=(.05, .16), ncol=2, frameon=False, fontsize=9)
    retention = ', '.join(f'{split} {mate["set_minus_frozen"][split]*100:+.2f} pp' for split in ('dev', 'shift'))
    fig.text(.055, .115, 'Mating-set ordinary-task retention versus frozen: '+retention+'.', fontsize=10)
    fig.text(.055, .065, 'Successor scoring uses exact native board transitions and learned value, not a learned world-model rollout. '
             'Compute panels reuse prior development positions.', fontsize=9, color='#506272')
    fig.text(.055, .03, 'Mate confirmation uses new source games from one cached public prefix. No Elo, RL or architectural novelty claim.',
             fontsize=9, color='#506272')
    return fig


def render(compute, mate, prefix, models=None):
    compute, mate, prefix = map(Path, (compute, mate, prefix))
    verified = {'compute': audit(compute), 'mate': audit(mate, models=models)}
    if verified['compute']['kind'] != 'compute' or verified['mate']['kind'] != 'mate':
        raise ValueError('Render inputs are not the specified compute and mate publications')
    paths = {extension: prefix.with_suffix('.'+extension) for extension in ('png', 'svg', 'json')}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Figure outputs already exist')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig = figure(read(compute/'summary.json'), read(mate/'summary.json'))
    fig.savefig(paths['png'], dpi=180)
    fig.savefig(paths['svg'], metadata={'Date': None})
    import matplotlib.pyplot as plt
    plt.close(fig)
    write_new(paths['json'], {'status': 'completed', 'verification': verified, 'publisher_sha256': sha(__file__),
                             'plots': {extension: sha(paths[extension]) for extension in ('png', 'svg')},
                             'source': 'Verified published summaries only; no new evaluation'})
    return {extension: str(path) for extension, path in paths.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    package = commands.add_parser('publish')
    package.add_argument('--kind', choices=('compute', 'mate'), required=True)
    for name in ('plan', 'execution', 'report', 'out'):
        package.add_argument('--'+name, type=Path, required=True)
    package.add_argument('--models', type=Path)
    check = commands.add_parser('audit')
    check.add_argument('--out', type=Path, required=True)
    check.add_argument('--models', type=Path)
    plot = commands.add_parser('render')
    for name in ('compute', 'mate', 'prefix'):
        plot.add_argument('--'+name, type=Path, required=True)
    plot.add_argument('--models', type=Path)
    args = vars(parser.parse_args())
    command = args.pop('command')
    print(json.dumps({'publish': publish, 'audit': audit, 'render': render}[command](**args), indent=2))


if __name__ == '__main__':
    main()
