"""Verify and losslessly package a completed spatial-chess development study.

Publication reuses the frozen report validator, without training, engine calls,
or model inference. Portable audit streams every archived file and checks its
receipt bindings without requiring the original engine binary or MPS host.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_spatial_publication_study',
                                             ROOT/'scripts/train_chess_spatial.py')
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)
MODES = tuple(STUDY.MODES)
SEEDS = tuple(STUDY.PROTOCOL['seeds'])
SPLITS = ('dev', 'shift')
RESULTS = Path('evidence/chess-spatial-v1/results')
WEIGHTS = Path('models/chess-spatial-v1')
ARCHIVE = 'execution-and-report.tar.gz'
METRICS = ('top1_teacher_agreement', 'target_nll', 'value_mae', 'future_changed_accuracy',
           'future_square_accuracy', 'future_exact_board_accuracy')
COLORS = {'cnn': '#7a899d', 'recurrent': '#7161a8', 'reconstruct': '#b58242', 'predict': '#16827d'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def decode_json(data):
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


def read_json(path):
    return decode_json(Path(path).read_bytes())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def close(actual, expected):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
        raise ValueError('Inconsistent summary aggregate')


def validate_summary(summary):
    if (summary['status'] != 'completed' or summary['protocol'] != STUDY.PROTOCOL
            or summary['novelty_established'] is not False or summary['elo_estimate'] is not None
            or summary['claim_scope'] != STUDY.PROTOCOL['claim_scope']):
        raise ValueError('Expected the scoped completed development summary')
    expected = {(mode, seed, split) for mode in MODES for seed in SEEDS for split in SPLITS}
    fits = summary['fits']
    if len(fits) != 24 or {(r['mode'], r['seed'], r['split']) for r in fits} != expected:
        raise ValueError('Summary requires all 12 fits on both evaluation splits')
    counts = {split['name']: split['examples'] for split in STUDY.PROTOCOL['splits']}
    for row in fits:
        if row['examples'] != counts[row['split']]:
            raise ValueError('Incorrect evaluation example count')
        for metric in METRICS:
            value = row[metric]
            if not math.isfinite(value) or value < 0 or (metric not in ('target_nll', 'value_mae') and value > 1):
                raise ValueError('Invalid fit metric')
        if row['value_mae'] > 2:
            raise ValueError('Invalid bounded value error')
    for split in SPLITS:
        for mode in MODES:
            for metric in METRICS:
                close(summary['means'][split][mode][metric], np.mean(
                    [r[metric] for r in fits if r['split'] == split and r['mode'] == mode]))
        greedy = summary['baselines'][split]['greedy_material']['metrics']['top1_teacher_agreement']
        if not math.isfinite(greedy) or not 0 <= greedy <= 1:
            raise ValueError('Invalid greedy baseline agreement')
    if summary['continuation_gate'] != STUDY.continuation_gate(fits):
        raise ValueError('Continuation gate differs from the frozen rule')
    return summary


def inventory(execution, report):
    """Include every regular file, rejecting links, special files and failures."""
    result = {}
    for prefix, directory in [('execution', Path(execution)), ('report', Path(report))]:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError('Evidence roots must be real directories')
        for path in sorted(directory.rglob('*')):
            if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                raise ValueError(f'Non-regular evidence artifact: {path}')
            if path.is_file():
                if path.name == 'failed.json':
                    raise ValueError('A failed receipt prevents publication')
                name = f'{prefix}/{path.relative_to(directory).as_posix()}'
                result[name] = {'sha256': sha(path), 'size': path.stat().st_size}
    return result


def _expected_names():
    return {f'{mode}-{seed}' for mode in MODES for seed in SEEDS}


def required_members():
    result = {'execution/started.json', 'execution/completed.json', 'execution/engine.json',
              'execution/baselines.json', 'report/summary.json', 'report/completed.json'}
    result.update(f'execution/data/{name}' for name in ('started.json', 'excluded-states.json',
                  'games.jsonl', 'analyses.jsonl', 'train.jsonl', 'dev.jsonl', 'shift.jsonl', 'completed.json'))
    result.update(f'execution/regret/{name}' for name in ('selection.json', 'analyses.jsonl',
                                                       'predictions.json', 'completed.json'))
    result.update(f'execution/{name}/{file}' for name in _expected_names()
                  for file in ('started.json', 'completed.json', 'weights.pt', 'learning.jsonl',
                               'dev.json', 'shift.json'))
    return result


def validate_chain(plan, plan_hash, summary, members, documents):
    """Check receipt coverage using the bytes actually recovered from the archive."""
    if not required_members() <= set(members):
        raise ValueError('Archive lacks required full-panel evidence')
    if plan['protocol'] != STUDY.PROTOCOL or summary['plan_sha256'] != plan_hash:
        raise ValueError('Plan and summary identities differ')
    complete = documents['execution/completed.json']
    started = documents['execution/started.json']
    report = documents['report/completed.json']
    expected = _expected_names()
    if (complete['status'] != 'completed' or complete['plan_sha256'] != plan_hash
            or started['plan_sha256'] != plan_hash or complete['fits'] != 12
            or set(complete['fit_receipts']) != expected or len(complete['fit_order']) != 12
            or {f'{mode}-{seed}' for mode, seed in complete['fit_order']} != expected
            or report['status'] != 'completed' or report['plan_sha256'] != plan_hash
            or report['execution_receipt_sha256'] != members['execution/completed.json']['sha256']
            or report['summary_sha256'] != members['report/summary.json']['sha256']):
        raise ValueError('Completion chain or full fit panel mismatch')
    if complete['baseline_sha256'] != members['execution/baselines.json']['sha256']:
        raise ValueError('Baseline receipt mismatch')
    if complete['regret_receipt_sha256'] != members['execution/regret/completed.json']['sha256']:
        raise ValueError('Regret receipt mismatch')
    data = documents['execution/data/completed.json']
    regret = documents['execution/regret/completed.json']
    if summary['teacher_cost'] != data or summary['regret'] != regret:
        raise ValueError('Summary cost receipts differ')
    for prefix, receipt in [('execution/data', data), ('execution/regret', regret)]:
        if receipt['status'] != 'completed':
            raise ValueError('Data or regret generation is incomplete')
        required = {name.removeprefix(prefix+'/') for name in required_members()
                    if name.startswith(prefix+'/') and not name.endswith('/completed.json')}
        if set(receipt['files']) != required:
            raise ValueError('Data or regret receipt coverage mismatch')
        for name, digest in receipt['files'].items():
            if members[f'{prefix}/{name}']['sha256'] != digest:
                raise ValueError('Archived receipt member hash mismatch')
    count = next(row['examples'] for row in STUDY.PROTOCOL['splits'] if row['name'] == 'train')
    updates = STUDY.PROTOCOL['epochs']*math.ceil(count/STUDY.PROTOCOL['batch_size'])
    if complete['total_updates'] != updates*12:
        raise ValueError('Training update budget differs')
    seconds = 0.
    for name in sorted(expected):
        prefix = f'execution/{name}'
        receipt = documents[prefix+'/completed.json']
        mode, seed_text = name.rsplit('-', 1)
        seed = int(seed_text)
        if (receipt['status'] != 'completed' or receipt['mode'] != mode or receipt['seed'] != seed
                or receipt['plan_sha256'] != plan_hash
                or receipt['data_receipt_sha256'] != members['execution/data/completed.json']['sha256']
                or complete['fit_receipts'][name] != members[prefix+'/completed.json']['sha256']
                or receipt['updates'] != updates or receipt['examples_seen'] != count*STUDY.PROTOCOL['epochs']
                or set(receipt['files']) != {'weights.pt', 'learning.jsonl', 'dev.json', 'shift.json'}):
            raise ValueError('Final fit identity or budget mismatch')
        for file, digest in receipt['files'].items():
            if members[f'{prefix}/{file}']['sha256'] != digest:
                raise ValueError('Archived fit artifact hash mismatch')
        for row in summary['fits']:
            if row['mode'] == mode and row['seed'] == seed:
                if (row['checkpoint_sha256'] != receipt['files']['weights.pt']
                        or row['parameter_count'] != receipt['parameter_count']
                        or row['parameter_count'] != plan['parameters'][mode]):
                    raise ValueError('Summary checkpoint identity mismatch')
                close(row['training_wall_seconds'], receipt['training_wall_seconds'])
        seconds += receipt['training_wall_seconds']
    close(complete['training_wall_seconds'], seconds)
    close(summary['training_wall_seconds'], seconds)


def make_archive(path, execution, report, members):
    """Fixed ordering, metadata and gzip timestamp; original payloads untouched."""
    roots = {'execution': Path(execution), 'report': Path(report)}
    with (Path(path).open('xb') as raw,
          gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed,
          tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive):
        for name, metadata in sorted(members.items()):
            prefix, relative = name.split('/', 1)
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = metadata['size'], 0o644, 0
            with (roots[prefix]/relative).open('rb') as stream:
                archive.addfile(info, stream)


def inspect_archive(path, expected):
    """Stream decompression, byte hashing and selected receipt parsing; no extraction."""
    actual, documents = {}, {}
    with tarfile.open(path, mode='r|gz') as archive:
        for member in archive:
            name = member.name
            components = PurePosixPath(name).parts
            if (not member.isfile() or name in actual or name not in expected
                    or not components or components[0] not in ('execution', 'report')
                    or '..' in components or str(PurePosixPath(name)) != name):
                raise ValueError('Unexpected, duplicated or unsafe archive member')
            if member.size != expected[name]['size']:
                raise ValueError('Archive member size mismatch')
            stream = archive.extractfile(member)
            digest = hashlib.sha256()
            keep = name.endswith('/completed.json') or name == 'execution/started.json'
            chunks = []
            while chunk := stream.read(1024*1024):
                digest.update(chunk)
                if keep:
                    chunks.append(chunk)
            actual[name] = {'sha256': digest.hexdigest(), 'size': member.size}
            if actual[name] != expected[name]:
                raise ValueError('Archive member hash mismatch')
            if keep:
                documents[name] = decode_json(b''.join(chunks))
    if actual != expected:
        raise ValueError('Archive membership differs from manifest')
    return actual, documents


def publish(plan_path, execution, report, project=ROOT):
    project, plan_path, execution, report = map(Path, (project, plan_path, execution, report))
    out, weights = project/RESULTS, project/WEIGHTS
    if out.exists() or weights.exists():
        raise FileExistsError('Publication or weights directory already exists')
    plan = STUDY.verify_plan(plan_path)
    summary = validate_summary(read_json(report/'summary.json'))
    report_receipt = read_json(report/'completed.json')
    if (report_receipt['summary_sha256'] != sha(report/'summary.json')
            or report_receipt['execution_receipt_sha256'] != sha(execution/'completed.json')):
        raise ValueError('Supplied report is not bound to its summary and execution')
    members = inventory(execution, report)
    if not required_members() <= set(members):
        raise ValueError('Missing required full-panel evidence')
    # Recompute from saved predictions/weights/logs, not from new model inference.
    with tempfile.TemporaryDirectory(prefix='openjev-spatial-publication-') as temporary:
        recomputed = STUDY.report(plan_path, execution, Path(temporary)/'report')
    if summary != recomputed:
        raise ValueError('Supplied summary does not reproduce from the frozen report')
    out.mkdir(parents=True, exist_ok=False)
    weights.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(plan_path, out/'plan.json')
    shutil.copyfile(report/'summary.json', out/'summary.json')
    make_archive(out/ARCHIVE, execution, report, members)
    if members != inventory(execution, report):
        raise ValueError('Evidence changed while publication was being prepared')
    weight_manifest = {}
    for name in sorted(_expected_names()):
        relative = WEIGHTS/name/'weights.pt'
        destination = project/relative
        destination.parent.mkdir()
        shutil.copyfile(execution/name/'weights.pt', destination)
        original = members[f'execution/{name}/weights.pt']
        if sha(destination) != original['sha256'] or destination.stat().st_size != original['size']:
            raise ValueError('Copied checkpoint differs from its original')
        weight_manifest[name] = {'path': relative.as_posix(), **original}
    manifest = {'version': 1, 'plan_sha256': sha(out/'plan.json'),
                'archive': {'path': ARCHIVE, 'sha256': sha(out/ARCHIVE), 'size': (out/ARCHIVE).stat().st_size},
                'members': members, 'member_count': len(members),
                'raw_bytes': sum(row['size'] for row in members.values()), 'weights': weight_manifest}
    write_new(out/'manifest.json', manifest)
    actual, documents = inspect_archive(out/ARCHIVE, members)
    validate_chain(plan, sha(out/'plan.json'), summary, actual, documents)
    write_new(out/'completed.json', {'status': 'completed', 'plan_sha256': sha(out/'plan.json'),
                                    'summary_sha256': sha(out/'summary.json'),
                                    'manifest_sha256': sha(out/'manifest.json'),
                                    'publisher_sha256': sha(__file__), 'fits': 12,
                                    'source_report_reproduced': True, 'archive_roundtrip_verified': True,
                                    'copied_weights_match_originals': True,
                                    'scope': 'Development evidence; no Elo or novelty claim'})
    return audit(project)


def audit(project=ROOT):
    """Portable publication integrity audit, without model inference or engine access."""
    project = Path(project)
    out = project/RESULTS
    receipt, manifest = read_json(out/'completed.json'), read_json(out/'manifest.json')
    summary = validate_summary(read_json(out/'summary.json'))
    plan_hash = sha(out/'plan.json')
    if (receipt['status'] != 'completed' or receipt['fits'] != 12
            or receipt['plan_sha256'] != plan_hash or manifest['plan_sha256'] != plan_hash
            or receipt['summary_sha256'] != sha(out/'summary.json')
            or receipt['manifest_sha256'] != sha(out/'manifest.json')
            or (out/'failed.json').exists()):
        raise ValueError('Publication completion receipt mismatch')
    archive = manifest['archive']
    if (archive['path'] != ARCHIVE or sha(out/ARCHIVE) != archive['sha256']
            or (out/ARCHIVE).stat().st_size != archive['size']):
        raise ValueError('Compressed archive checksum mismatch')
    members, documents = inspect_archive(out/ARCHIVE, manifest['members'])
    if (manifest['member_count'] != len(members)
            or manifest['raw_bytes'] != sum(row['size'] for row in members.values())
            or members['report/summary.json']['sha256'] != sha(out/'summary.json')
            or set(manifest['weights']) != _expected_names()):
        raise ValueError('Manifest totals or required checkpoint coverage mismatch')
    validate_chain(read_json(out/'plan.json'), plan_hash, summary, members, documents)
    for name, row in manifest['weights'].items():
        relative = WEIGHTS/name/'weights.pt'
        path = project/relative
        original = members[f'execution/{name}/weights.pt']
        if (row['path'] != relative.as_posix() or row['sha256'] != original['sha256']
                or row['size'] != original['size'] or path.is_symlink()
                or sha(path) != original['sha256'] or path.stat().st_size != original['size']):
            raise ValueError('Published checkpoint differs from archived original')
    return {'status': 'verified', 'fits': 12, 'members': len(members), 'plan_sha256': plan_hash,
            'archive_sha256': archive['sha256'], 'summary_sha256': sha(out/'summary.json')}


def figure(summary):
    validate_summary(summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.hashsalt': 'chess-spatial-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), gridspec_kw={'width_ratios': [1.05, 1]})
    fig.patch.set_facecolor('#fbfcfe')
    fig.subplots_adjust(left=.065, right=.975, top=.75, bottom=.27, wspace=.28)
    fig.text(.065, .93, 'OpenJev spatial chess study', fontsize=21, weight='bold', color='#192b3c')
    fig.text(.065, .865, 'Final checkpoints | 3 training seeds per model | development evaluation',
             fontsize=11, color='#536272')
    fit_map = {(row['mode'], row['seed'], row['split']): row for row in summary['fits']}
    labels = {'cnn': 'CNN', 'recurrent': 'Recurrent', 'reconstruct': 'Reconstruct', 'predict': 'Predict'}
    for ax in axes:
        ax.set_facecolor('#fbfcfe')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
    for offset, split, marker in [(-.18, 'dev', 'o'), (.18, 'shift', 's')]:
        for x, mode in enumerate(MODES):
            values = [fit_map[mode, seed, split]['top1_teacher_agreement']*100 for seed in SEEDS]
            axes[0].bar(x+offset, np.mean(values), width=.32, color=COLORS[mode],
                        alpha=1 if split == 'dev' else .5, zorder=2)
            axes[0].scatter(x+offset+np.array([-.065, 0, .065]), values, marker=marker,
                            s=24, facecolor='white', edgecolor='#223849', zorder=3)
        baseline = summary['baselines'][split]['greedy_material']['metrics']['top1_teacher_agreement']*100
        axes[0].axhline(baseline, color='#a6513e', linestyle='-' if split == 'dev' else '--', linewidth=1)
    axes[0].set_xticks(range(4), [labels[mode] for mode in MODES])
    axes[0].set_title('Stockfish top-move agreement', loc='left', fontsize=11)
    peak = max([row['top1_teacher_agreement']*100 for row in summary['fits']]
               + [summary['baselines'][split]['greedy_material']['metrics']['top1_teacher_agreement']*100
                  for split in SPLITS])
    axes[0].set_ylim(0, min(100, max(20, peak*1.2)))
    groups = [('reconstruct', 'future_changed_accuracy'), ('predict', 'future_changed_accuracy'),
              ('reconstruct', 'future_exact_board_accuracy'), ('predict', 'future_exact_board_accuracy')]
    for x, (mode, metric) in enumerate(groups):
        for offset, split, marker in [(-.18, 'dev', 'o'), (.18, 'shift', 's')]:
            values = [fit_map[mode, seed, split][metric]*100 for seed in SEEDS]
            axes[1].bar(x+offset, np.mean(values), width=.32, color=COLORS[mode],
                        alpha=1 if split == 'dev' else .5, zorder=2)
            axes[1].scatter(x+offset+np.array([-.065, 0, .065]), values, marker=marker,
                            s=24, facecolor='white', edgecolor='#223849', zorder=3)
    axes[1].set_xticks(range(4), ['Reconstruct\nchanged squares', 'Predict\nchanged squares',
                                'Reconstruct\nexact board', 'Predict\nexact board'], fontsize=9)
    axes[1].set_title('Action-conditioned future-piece prediction', loc='left', fontsize=11)
    handles = [Line2D([], [], color='#344b60', marker='o', label='Development', linestyle=''),
               Line2D([], [], color='#344b60', marker='s', label='Shifted development', linestyle=''),
               Line2D([], [], color='#a6513e', label='Greedy: development', linestyle='-'),
               Line2D([], [], color='#a6513e', label='Greedy: shifted', linestyle='--')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.06, .115), ncol=4, frameon=False, fontsize=9)
    status = 'PASSED' if summary['continuation_gate']['passed'] else 'FAILED'
    fig.text(.065, .08, f'Frozen development gate: {status}. Dots show every seed; bars show means.', fontsize=10)
    fig.text(.065, .035, 'Reconstruct predicts the current board during training; both decoders are scored on '
             'future boards here. No Elo or novelty claim.', fontsize=9, color='#536272')
    return fig


def render(project=ROOT, prefix=None):
    project = Path(project)
    verification = audit(project)
    prefix = project/'docs/assets/chess-spatial-results' if prefix is None else Path(prefix)
    paths = {extension: prefix.with_suffix('.'+extension) for extension in ('png', 'svg', 'json')}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Figure output already exists')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig = figure(read_json(project/RESULTS/'summary.json'))
    fig.savefig(paths['png'], dpi=180)
    fig.savefig(paths['svg'], metadata={'Date': None})
    import matplotlib.pyplot as plt
    plt.close(fig)
    write_new(paths['json'], {'status': 'completed', 'verification': verification,
                             'publisher_sha256': sha(__file__),
                             'plots': {ext: sha(paths[ext]) for ext in ('png', 'svg')}})
    return {extension: str(path) for extension, path in paths.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    package = commands.add_parser('publish')
    for name in ('plan', 'execution', 'report'):
        package.add_argument('--'+name, type=Path, required=True)
    package.add_argument('--project', type=Path, default=ROOT)
    for command in ('audit', 'render'):
        sub = commands.add_parser(command)
        sub.add_argument('--project', type=Path, default=ROOT)
        if command == 'render':
            sub.add_argument('--prefix', type=Path)
    args = parser.parse_args()
    if args.command == 'publish':
        result = publish(args.plan, args.execution, args.report, args.project)
    elif args.command == 'audit':
        result = audit(args.project)
    else:
        result = render(args.project, args.prefix)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
